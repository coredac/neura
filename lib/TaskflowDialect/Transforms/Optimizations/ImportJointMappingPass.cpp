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
// SCOPE:
//   v1 ("tasks" schema): per existing taskflow.task, the JSON assigns a
//       placement + shape. Task bodies are left untouched (no fusion).
//   v2 ("groups" schema): the JSON also carries a task-FUSION decision. Each
//       group lists one or more task names; a group with >1 task is first
//       MERGED into a single task (reusing the tested resource-aware fusion --
//       see fuseTaskGroup / UtilizationFuser::performFusion), then the shape /
//       placement / context attributes are written onto the resulting (fused)
//       task. Single-task groups are untouched apart from the attribute write.
//
// The two schemas are mutually exclusive and auto-detected: a top-level
// "groups" array selects v2; otherwise the v1 "tasks" path runs unchanged.
//
// JSON schema v1 (top-level object with a "tasks" array; a bare top-level array
// is also accepted):
//   {
//     "tasks": [
//       { "task": "Task_0", "cgra_count": 1, "cgra_shape": "1x1",
//         "row": 0, "col": 0, "context_id": 0 },
//       ...
//     ]
//   }
//
// JSON schema v2 (top-level object with a "groups" array):
//   {
//     "groups": [
//       { "tasks": ["Task_0", "Task_2"], "cgra_count": 1, "cgra_shape": "1x1",
//         "row": 0, "col": 0, "context_id": 0 },
//       { "tasks": ["Task_1"], "cgra_count": 1, "row": 0, "col": 1,
//         "context_id": 0 }
//     ]
//   }
//
// For each task (v1) or resulting group task (v2) this pass:
//   1. sets cgra_count (i32) and cgra_shape (string) on the task op
//      (same attributes ResourceAwareTaskOptimizationPass writes);
//   2. sets task_orchestration_info = { cgra_positions = [{row, col,
//      context_id}] } (same nested-attribute structure orchestration_utils.cpp
//      builds).
//
// It therefore REPLACES orchestrate-tasks-on-accelerators (placement + temporal
// context assignment), overrides resource-aware's shape choice with the
// imported one, and (v2) replaces resource-aware's fusion decision.

#include "TaskflowDialect/TaskFusionUtil.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>

using namespace mlir;
using namespace mlir::taskflow;

namespace {

// One imported placement/shape decision.
struct ImportedTaskMapping {
  int cgra_count = 1;
  std::string cgra_shape = "1x1";
  int row = 0;
  int col = 0;
  int context_id = 0;
};

// One imported v2 group: a set of task names + the decision applied to the
// (possibly fused) resulting task.
struct ImportedGroup {
  SmallVector<std::string> tasks; // >= 1 member.
  ImportedTaskMapping mapping;
};

struct ImportJointMappingPass
    : public PassWrapper<ImportJointMappingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ImportJointMappingPass)

  ImportJointMappingPass() = default;
  ImportJointMappingPass(const ImportJointMappingPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override { return "import-joint-mapping"; }
  StringRef getDescription() const override {
    return "Replays an externally-computed joint multi-task mapping (optionally "
           "including task fusion) onto Taskflow tasks, bypassing the staged "
           "greedy orchestrator (--import-joint-mapping=<file.json>).";
  }

  // Path to the JSON decision file. The mlir-neura-opt driver also accepts the
  // shorthand --import-joint-mapping=<file.json> and rewrites it to
  // --import-joint-mapping=file=<file.json> before the pass parser runs.
  Option<std::string> mappingFile{
      *this, "file",
      llvm::cl::desc("Path to the JSON joint-mapping decision file."),
      llvm::cl::init("")};

  //===--------------------------------------------------------------------===//
  // Attribute writing (shared by v1 and v2)
  //===--------------------------------------------------------------------===//

  // Writes the resource-binding + orchestration attributes onto `task`, exactly
  // as ResourceAwareTaskOptimizationPass and orchestration_utils.cpp do.
  static void applyMapping(TaskflowTaskOp task, const ImportedTaskMapping &m) {
    MLIRContext *ctx = task.getContext();
    OpBuilder builder(ctx);

    // 1. Resource-binding attributes (override resource-aware's choice).
    task->setAttr("cgra_count", builder.getI32IntegerAttr(m.cgra_count));
    task->setAttr("cgra_shape", builder.getStringAttr(m.cgra_shape));

    // 2. task_orchestration_info = { cgra_positions = [{row, col, context_id}] }
    //    Built with the exact same builder calls as orchestration_utils.cpp;
    //    DictionaryAttr keys must be alphabetical (col < context_id < row).
    SmallVector<NamedAttribute, 3> coord_attrs;
    coord_attrs.push_back(NamedAttribute(StringAttr::get(ctx, "col"),
                                         builder.getI32IntegerAttr(m.col)));
    coord_attrs.push_back(
        NamedAttribute(StringAttr::get(ctx, "context_id"),
                       builder.getI32IntegerAttr(m.context_id)));
    coord_attrs.push_back(NamedAttribute(StringAttr::get(ctx, "row"),
                                         builder.getI32IntegerAttr(m.row)));

    SmallVector<Attribute> pos_attrs;
    pos_attrs.push_back(DictionaryAttr::get(ctx, coord_attrs));

    SmallVector<NamedAttribute, 1> mapping_attrs;
    mapping_attrs.push_back(
        NamedAttribute(StringAttr::get(ctx, "cgra_positions"),
                       builder.getArrayAttr(pos_attrs)));
    task->setAttr("task_orchestration_info",
                  DictionaryAttr::get(ctx, mapping_attrs));
  }

  //===--------------------------------------------------------------------===//
  // JSON parsing
  //===--------------------------------------------------------------------===//

  // Reads the shared placement/shape fields from a JSON record into `m`.
  // row/col are required; the rest default (context_id=0, cgra_count=1,
  // cgra_shape="1x1").
  static bool parseMappingFields(const llvm::json::Object &record,
                                 ImportedTaskMapping &m, std::string &err) {
    std::optional<int64_t> row = record.getInteger("row");
    std::optional<int64_t> col = record.getInteger("col");
    if (!row || !col) {
      err = "entry missing required \"row\"/\"col\"";
      return false;
    }
    m.row = static_cast<int>(*row);
    m.col = static_cast<int>(*col);
    m.context_id = static_cast<int>(record.getInteger("context_id").value_or(0));
    m.cgra_count = static_cast<int>(record.getInteger("cgra_count").value_or(1));
    if (std::optional<StringRef> shape = record.getString("cgra_shape"))
      m.cgra_shape = shape->str();
    return true;
  }

  // Parses the v1 "tasks" array (or bare top-level array) into a name -> mapping
  // table.
  static bool parseTasks(llvm::json::Array &tasks,
                         llvm::StringMap<ImportedTaskMapping> &out,
                         std::string &err) {
    for (llvm::json::Value &entry : tasks) {
      llvm::json::Object *record = entry.getAsObject();
      if (!record) {
        err = "task entry is not an object";
        return false;
      }
      std::optional<StringRef> name = record->getString("task");
      if (!name) {
        err = "task entry missing required \"task\"";
        return false;
      }
      ImportedTaskMapping m;
      if (!parseMappingFields(*record, m, err))
        return false;
      if (out.count(*name)) {
        err = "duplicate task entry \"" + name->str() + "\"";
        return false;
      }
      out[*name] = m;
    }
    return true;
  }

  // Parses the v2 "groups" array into a list of ImportedGroup.
  static bool parseGroups(llvm::json::Array &groups,
                          SmallVector<ImportedGroup> &out, std::string &err) {
    for (llvm::json::Value &entry : groups) {
      llvm::json::Object *record = entry.getAsObject();
      if (!record) {
        err = "group entry is not an object";
        return false;
      }
      llvm::json::Array *tasks = record->getArray("tasks");
      if (!tasks || tasks->empty()) {
        err = "group entry missing a non-empty \"tasks\" array";
        return false;
      }
      ImportedGroup grp;
      for (llvm::json::Value &t : *tasks) {
        std::optional<StringRef> nm = t.getAsString();
        if (!nm) {
          err = "group \"tasks\" entry is not a string";
          return false;
        }
        grp.tasks.push_back(nm->str());
      }
      if (!parseMappingFields(*record, grp.mapping, err))
        return false;
      out.push_back(std::move(grp));
    }
    return true;
  }

  // Loads and dispatches on schema. Fills either `taskMap` (v1) or `groups`
  // (v2, and sets has_groups=true).
  static bool parseMappingFile(StringRef path,
                               llvm::StringMap<ImportedTaskMapping> &taskMap,
                               SmallVector<ImportedGroup> &groups,
                               bool &has_groups, std::string &err) {
    has_groups = false;
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

    if (llvm::json::Object *root = parsed->getAsObject()) {
      // v2 takes precedence when a "groups" array is present.
      if (llvm::json::Array *g = root->getArray("groups")) {
        has_groups = true;
        return parseGroups(*g, groups, err);
      }
      llvm::json::Array *tasks = root->getArray("tasks");
      if (!tasks) {
        err = "top-level object is missing a \"tasks\" or \"groups\" array";
        return false;
      }
      return parseTasks(*tasks, taskMap, err);
    }
    if (llvm::json::Array *arr = parsed->getAsArray())
      return parseTasks(*arr, taskMap, err);

    err = "top-level JSON must be an object with \"tasks\"/\"groups\" or an "
          "array";
    return false;
  }

  //===--------------------------------------------------------------------===//
  // v1: per-task replay (no fusion)
  //===--------------------------------------------------------------------===//

  void runV1(func::FuncOp func,
             const llvm::StringMap<ImportedTaskMapping> &mapping) {
    llvm::DenseSet<StringRef> applied;
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
      applied.insert(it->first());
      applyMapping(task, it->second);
    });

    if (!has_task) {
      func.emitWarning()
          << "[import-joint-mapping] function has no taskflow.task ops";
    }

    // Report imported entries that never matched a task (catches JSON typos).
    for (auto &kv : mapping) {
      if (!applied.contains(kv.first())) {
        func.emitWarning()
            << "[import-joint-mapping] imported entry for task \""
            << kv.first() << "\" did not match any taskflow.task";
      }
    }
  }

  //===--------------------------------------------------------------------===//
  // v2: fuse-then-place replay
  //===--------------------------------------------------------------------===//

  // Checks that the imported groups COVER the function's tasks, BEFORE any of
  // them is rewritten. The groups must partition the task set: every existing
  // task is named by exactly one group, and every name a group carries exists.
  //
  // This replaces a check that could not fail. The old one ran after the
  // replay and asked whether each surviving task carried a "cgra_shape"
  // attribute -- but cgra_shape is precisely what
  // ResourceAwareTaskOptimizationPass writes, and this pass is documented to
  // run on that output ("overrides resource-aware's shape choice with the
  // imported one"). Every task therefore already carried the attribute before
  // the import started, so the walk saw "covered" for tasks no group had
  // mentioned; an import covering NOTHING ("groups": []) was reported as a
  // success with not one placement written. Coverage is now decided by the
  // imported names themselves, which no earlier pass can pre-satisfy.
  //
  // Validating up front also means an incomplete decision is REFUSED rather
  // than half-applied: the pass replays an external solver's joint decision so
  // its cost can be measured, and a decision that is only partly replayed is
  // not the decision that was solved for. Checking first keeps the IR from
  // being fused/annotated on the way to the error.
  //
  // The partition property is what later lets coverage be re-checked by op
  // identity: groups with disjoint task names fuse disjoint sets of ops, so no
  // group can consume the task another group already placed.
  bool validateGroupCoverage(func::FuncOp func,
                             const SmallVector<ImportedGroup> &groups) {
    bool valid = true;

    // 1. No task may be claimed by two groups (that would be two different
    //    fusion/placement decisions for one task).
    llvm::StringSet<> imported_names;
    for (const ImportedGroup &grp : groups) {
      for (const std::string &name : grp.tasks) {
        if (!imported_names.insert(name).second) {
          func.emitError() << "[import-joint-mapping] task \"" << name
                           << "\" appears in more than one imported group";
          valid = false;
        }
      }
    }

    // 2. Every task the function actually has must be named by some group.
    llvm::StringSet<> tasks_in_ir;
    func.walk([&](TaskflowTaskOp task) {
      tasks_in_ir.insert(task.getTaskName());
      if (!imported_names.contains(task.getTaskName())) {
        task.emitError() << "[import-joint-mapping] task \""
                         << task.getTaskName()
                         << "\" is not covered by any imported group";
        valid = false;
      }
    });

    // 3. Every name a group carries must exist (a typo, or a JSON produced from
    //    different IR, otherwise silently drops that task's decision). Walked in
    //    group order so the diagnostics are deterministic.
    for (const ImportedGroup &grp : groups) {
      for (const std::string &name : grp.tasks) {
        if (!tasks_in_ir.contains(name)) {
          func.emitError() << "[import-joint-mapping] imported group names "
                              "task \""
                           << name
                           << "\", which does not exist in this function";
          valid = false;
        }
      }
    }
    return valid;
  }

  void runV2(func::FuncOp func, const SmallVector<ImportedGroup> &groups) {
    if (!validateGroupCoverage(func, groups)) {
      signalPassFailure();
      return;
    }

    // Tasks the replay actually placed, tracked by identity so the post-check
    // below cannot be satisfied by an attribute an earlier pass left behind.
    llvm::DenseSet<Operation *> placed_tasks;

    for (const ImportedGroup &grp : groups) {
      TaskflowTaskOp target;

      if (grp.tasks.size() == 1) {
        // Singleton group: locate the task, leave its body untouched.
        StringRef name = grp.tasks.front();
        func.walk([&](TaskflowTaskOp t) {
          if (t.getTaskName() == name)
            target = t;
        });
        if (!target) {
          func.emitError() << "[import-joint-mapping] group task \""
                           << grp.tasks.front()
                           << "\" did not match any taskflow.task";
          signalPassFailure();
          return;
        }
      } else {
        // Multi-task group: MERGE via the shared, dominance-safe fusion helper
        // (reuses UtilizationFuser::performFusion). Reject cleanly on any
        // illegal/unsafe group instead of emitting invalid IR.
        std::string ferr;
        FailureOr<TaskflowTaskOp> fused =
            fuseTaskGroup(func, grp.tasks, /*analytical=*/true, ferr);
        if (failed(fused)) {
          func.emitError() << "[import-joint-mapping] " << ferr;
          signalPassFailure();
          return;
        }
        target = *fused;
      }

      applyMapping(target, grp.mapping);
      placed_tasks.insert(target.getOperation());
    }

    // Post-check: every task LEFT IN THE IR was placed by this replay. The
    // pre-check above already proved the imported names cover the tasks that
    // existed on entry; this catches a task the rewriting itself produced or
    // left behind (e.g. a group whose fusion did not consume every member), and
    // it asks by op identity, not by an attribute any earlier pass may have
    // written.
    bool all_covered = true;
    func.walk([&](TaskflowTaskOp task) {
      if (!placed_tasks.contains(task.getOperation())) {
        task.emitError() << "[import-joint-mapping] task \""
                         << task.getTaskName()
                         << "\" survived the import without a placement (not "
                            "covered by any imported group)";
        all_covered = false;
      }
    });
    if (!all_covered)
      signalPassFailure();
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

    llvm::StringMap<ImportedTaskMapping> taskMap;
    SmallVector<ImportedGroup> groups;
    bool has_groups = false;
    std::string err;
    if (!parseMappingFile(mappingFile.getValue(), taskMap, groups, has_groups,
                          err)) {
      func.emitError() << "[import-joint-mapping] " << err;
      signalPassFailure();
      return;
    }

    if (has_groups)
      runV2(func, groups);
    else
      runV1(func, taskMap);
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
