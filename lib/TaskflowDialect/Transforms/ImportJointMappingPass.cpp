// Import an externally-computed JOINT multi-task mapping and replay it onto the
// Taskflow tasks, bypassing the staged greedy orchestrator.
//
// This pass is the joint-mapping analogue of MapToAcceleratorPass's
// --import-mapping option: instead of running the internal placer/scheduler
// (orchestrate-tasks-on-accelerators), it reads a JSON decision file and writes
// exactly the IR attributes that the real orchestrate pass writes, so the
// downstream codegen is unchanged -- but the placement/shape decision comes
// from the file, not the greedy scheduler.
//
// SCOPE v1: per existing taskflow.task, the JSON assigns a placement + shape.
// It does NOT re-implement task fusion (that is v2); the task bodies are left
// untouched.
//
// JSON schema (top-level object with a "tasks" array; a bare top-level array is
// also accepted):
//   {
//     "tasks": [
//       { "task": "Task_0", "cgra_count": 1, "cgra_shape": "1x1",
//         "row": 0, "col": 0, "context_id": 0 },
//       ...
//     ]
//   }
//
// For each taskflow.task @Name this pass:
//   1. sets cgra_count (i32) and cgra_shape (string) on the task op
//      (same attributes ResourceAwareTaskOptimizationPass writes);
//   2. sets task_orchestration_info = { cgra_positions = [{row, col,
//      context_id}] } (same nested-attribute structure orchestration_utils.cpp
//      builds);
//   3. leaves the task body untouched.
//
// It therefore REPLACES orchestrate-tasks-on-accelerators (placement + temporal
// context assignment) and overrides resource-aware's shape choice with the
// imported one.

#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>

using namespace mlir;
using namespace mlir::taskflow;

namespace {

// One imported per-task decision.
struct ImportedTaskMapping {
  int cgra_count = 1;
  std::string cgra_shape = "1x1";
  int row = 0;
  int col = 0;
  int context_id = 0;
};

struct ImportJointMappingPass
    : public PassWrapper<ImportJointMappingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ImportJointMappingPass)

  ImportJointMappingPass() = default;
  ImportJointMappingPass(const ImportJointMappingPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override { return "import-joint-mapping"; }
  StringRef getDescription() const override {
    return "Replays an externally-computed joint multi-task mapping onto "
           "Taskflow tasks, bypassing the staged greedy orchestrator "
           "(--import-joint-mapping=<file.json>).";
  }

  // Path to the JSON decision file. The mlir-neura-opt driver also accepts the
  // shorthand --import-joint-mapping=<file.json> and rewrites it to
  // --import-joint-mapping=file=<file.json> before the pass parser runs.
  Option<std::string> mappingFile{
      *this, "file",
      llvm::cl::desc("Path to the JSON joint-mapping decision file."),
      llvm::cl::init("")};

  // Parses the JSON decision file into a name -> mapping table.
  static bool parseMappingFile(StringRef path,
                               llvm::StringMap<ImportedTaskMapping> &out,
                               std::string &err) {
    auto buffer = llvm::MemoryBuffer::getFile(path);
    if (!buffer) {
      err = "cannot open " + path.str();
      return false;
    }
    llvm::Expected<llvm::json::Value> parsed =
        llvm::json::parse((*buffer)->getBuffer());
    if (!parsed) {
      err = "invalid JSON: " + llvm::toString(parsed.takeError());
      return false;
    }
    // Accept either a top-level object with a "tasks" array, or a bare
    // top-level array of task entries.
    llvm::json::Array *tasks = nullptr;
    if (llvm::json::Object *root = parsed->getAsObject()) {
      tasks = root->getArray("tasks");
      if (!tasks) {
        err = "top-level object is missing a \"tasks\" array";
        return false;
      }
    } else if (llvm::json::Array *arr = parsed->getAsArray()) {
      tasks = arr;
    } else {
      err = "top-level JSON must be an object with \"tasks\" or an array";
      return false;
    }

    for (llvm::json::Value &entry : *tasks) {
      llvm::json::Object *record = entry.getAsObject();
      if (!record) {
        err = "task entry is not an object";
        return false;
      }
      std::optional<StringRef> name = record->getString("task");
      std::optional<int64_t> row = record->getInteger("row");
      std::optional<int64_t> col = record->getInteger("col");
      if (!name || !row || !col) {
        err = "task entry missing required \"task\"/\"row\"/\"col\"";
        return false;
      }
      ImportedTaskMapping m;
      m.row = static_cast<int>(*row);
      m.col = static_cast<int>(*col);
      // context_id defaults to 0 (spatial-only placement).
      m.context_id = static_cast<int>(record->getInteger("context_id").value_or(0));
      // cgra_count / cgra_shape default to a single 1x1 CGRA.
      m.cgra_count = static_cast<int>(record->getInteger("cgra_count").value_or(1));
      if (std::optional<StringRef> shape = record->getString("cgra_shape")) {
        m.cgra_shape = shape->str();
      }
      if (out.count(*name)) {
        err = "duplicate task entry \"" + name->str() + "\"";
        return false;
      }
      out[*name] = m;
    }
    return true;
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    if (mappingFile.getValue().empty()) {
      func.emitError()
          << "[import-joint-mapping] no mapping file provided; use "
             "--import-joint-mapping=<file.json>";
      signalPassFailure();
      return;
    }

    llvm::StringMap<ImportedTaskMapping> mapping;
    std::string err;
    if (!parseMappingFile(mappingFile.getValue(), mapping, err)) {
      func.emitError() << "[import-joint-mapping] " << err;
      signalPassFailure();
      return;
    }

    MLIRContext *ctx = func.getContext();
    OpBuilder builder(ctx);
    llvm::DenseSet<StringRef> applied;

    // For every task in the function, apply its imported decision.
    bool has_task = false;
    func.walk([&](TaskflowTaskOp task) {
      has_task = true;
      StringRef task_name = task.getTaskName();
      auto it = mapping.find(task_name);
      if (it == mapping.end()) {
        task.emitError() << "[import-joint-mapping] no imported mapping for "
                            "task \""
                         << task_name << "\"";
        signalPassFailure();
        return;
      }
      const ImportedTaskMapping &m = it->second;
      applied.insert(it->first());

      // 1. Resource-binding attributes (override resource-aware's choice).
      task->setAttr("cgra_count", builder.getI32IntegerAttr(m.cgra_count));
      task->setAttr("cgra_shape", builder.getStringAttr(m.cgra_shape));

      // 2. task_orchestration_info = { cgra_positions = [{row, col,
      //    context_id}] }. Built with the exact same builder calls as
      //    orchestration_utils.cpp; DictionaryAttr keys must be alphabetical
      //    (col < context_id < row).
      SmallVector<NamedAttribute, 3> coord_attrs;
      coord_attrs.push_back(NamedAttribute(
          StringAttr::get(ctx, "col"), builder.getI32IntegerAttr(m.col)));
      coord_attrs.push_back(
          NamedAttribute(StringAttr::get(ctx, "context_id"),
                         builder.getI32IntegerAttr(m.context_id)));
      coord_attrs.push_back(NamedAttribute(
          StringAttr::get(ctx, "row"), builder.getI32IntegerAttr(m.row)));

      SmallVector<Attribute> pos_attrs;
      pos_attrs.push_back(DictionaryAttr::get(ctx, coord_attrs));

      SmallVector<NamedAttribute, 1> mapping_attrs;
      mapping_attrs.push_back(
          NamedAttribute(StringAttr::get(ctx, "cgra_positions"),
                         builder.getArrayAttr(pos_attrs)));
      task->setAttr("task_orchestration_info",
                    DictionaryAttr::get(ctx, mapping_attrs));

      // 3. Task body is left untouched.
    });

    if (!has_task) {
      func.emitWarning()
          << "[import-joint-mapping] function has no taskflow.task ops";
    }

    // Report imported entries that never matched a task (helps catch typos in
    // the JSON), but do not fail the pass on them.
    for (auto &kv : mapping) {
      if (!applied.contains(kv.first())) {
        func.emitWarning()
            << "[import-joint-mapping] imported entry for task \""
            << kv.first() << "\" did not match any taskflow.task";
      }
    }
  }
};

} // namespace

namespace mlir {
namespace taskflow {

std::unique_ptr<Pass> createImportJointMappingPass() {
  return std::make_unique<ImportJointMappingPass>();
}

} // namespace taskflow
} // namespace mlir
