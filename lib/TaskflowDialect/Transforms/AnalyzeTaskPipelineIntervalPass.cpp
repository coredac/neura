// Analyze scheduled Taskflow pipeline interval.

#include "TaskflowDialect/Orchestration/orchestration_utils.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cstdint>
#include <optional>

using namespace mlir;
using namespace mlir::taskflow;

namespace {

static int64_t getRequiredTaskDuration(TaskflowTaskOp task) {
  // Prefer `est_latency` = II*(trip_count-1) + steps when the resource-aware
  // pass published it. It is the same quantity TaskScheduler charged for
  // residency when it built this placement; analysing the placement with
  // `profile_info.duration` (the DFG depth) instead would measure a different
  // schedule from the one that was actually produced.
  if (auto est_latency_attr =
          task->getAttrOfType<IntegerAttr>("est_latency")) {
    int64_t est_latency_cycles = est_latency_attr.getInt();
    if (est_latency_cycles > 0)
      return est_latency_cycles;
  }
  auto profile_info = task->getAttrOfType<DictionaryAttr>("profile_info");
  if (!profile_info) {
    task.emitOpError() << "requires profile_info.duration";
    return -1;
  }

  auto duration = dyn_cast_or_null<IntegerAttr>(profile_info.get("duration"));
  if (!duration) {
    task.emitOpError() << "requires profile_info.duration";
    return -1;
  }

  return std::max<int64_t>(1, duration.getInt());
}

static std::optional<TaskScheduleResult::CgraOccupancy>
parseCgraOccupancy(TaskflowTaskOp task, Attribute attr) {
  auto coord = dyn_cast<DictionaryAttr>(attr);
  if (!coord) {
    task.emitOpError() << "requires dictionary entries in "
                          "task_orchestration_info.cgra_positions";
    return std::nullopt;
  }

  auto row = dyn_cast_or_null<IntegerAttr>(coord.get("row"));
  auto col = dyn_cast_or_null<IntegerAttr>(coord.get("col"));
  auto context_id = dyn_cast_or_null<IntegerAttr>(coord.get("context_id"));
  if (!row || !col || !context_id) {
    task.emitOpError() << "requires row, col, and context_id in each "
                          "task_orchestration_info.cgra_positions entry";
    return std::nullopt;
  }

  TaskScheduleResult::CgraOccupancy occupancy;
  occupancy.row = static_cast<int>(row.getInt());
  occupancy.col = static_cast<int>(col.getInt());
  occupancy.context_id = static_cast<int>(context_id.getInt());
  return occupancy;
}

static LogicalResult readTaskCgraOccupancies(
    TaskflowTaskOp task,
    SmallVectorImpl<TaskScheduleResult::CgraOccupancy> &occupancies) {
  auto orchestration_info =
      task->getAttrOfType<DictionaryAttr>("task_orchestration_info");
  if (!orchestration_info) {
    return task.emitOpError() << "requires task_orchestration_info";
  }

  auto cgra_positions =
      dyn_cast_or_null<ArrayAttr>(orchestration_info.get("cgra_positions"));
  if (!cgra_positions || cgra_positions.empty()) {
    return task.emitOpError()
           << "requires non-empty task_orchestration_info.cgra_positions";
  }

  for (Attribute attr : cgra_positions) {
    std::optional<TaskScheduleResult::CgraOccupancy> occupancy =
        parseCgraOccupancy(task, attr);
    if (!occupancy) {
      return failure();
    }
    occupancies.push_back(*occupancy);
  }
  return success();
}

static int64_t
encodeCgraLocation(const TaskScheduleResult::CgraOccupancy &occupancy) {
  return (static_cast<int64_t>(occupancy.row) << 32) |
         static_cast<uint32_t>(occupancy.col);
}

class TaskScheduleResultParser {
public:
  explicit TaskScheduleResultParser(func::FuncOp func) : func_(func) {}

  FailureOr<SmallVector<TaskScheduleResult>> parse() {
    collectTasks();
    if (failed(buildScheduleResults())) {
      return failure();
    }
    buildTaskDependences();
    buildCgraExecutionOrderDependences();
    if (failed(computeStartTimes())) {
      return failure();
    }
    updateScheduleTimes();
    return schedule_result_;
  }

private:
  void collectTasks() {
    func_.walk([&](TaskflowTaskOp task) {
      task_to_index_[task.getOperation()] = tasks_.size();
      tasks_.push_back(task);
    });
  }

  LogicalResult buildScheduleResults() {
    for (TaskflowTaskOp task : tasks_) {
      int64_t duration = getRequiredTaskDuration(task);
      if (duration < 0) {
        return failure();
      }

      TaskScheduleResult task_result;
      task_result.task = task;
      task_result.duration = duration;
      task_result.end_time = duration;
      if (failed(readTaskCgraOccupancies(task, task_result.cgra_occupancies))) {
        return failure();
      }

      for (TaskScheduleResult::CgraOccupancy &occupancy :
           task_result.cgra_occupancies) {
        occupancy.duration = duration;
      }

      schedule_result_.push_back(std::move(task_result));
    }
    return success();
  }

  void addExecutionOrderDependence(int task, int next_task) {
    if (task < 0 || next_task < 0 || task == next_task) {
      return;
    }

    int64_t edge_key =
        (static_cast<int64_t>(task) << 32) | static_cast<uint32_t>(next_task);
    if (!execution_order_edge_keys_.insert(edge_key).second) {
      return;
    }

    successors_[task].push_back(next_task);
    ++predecessor_count_[next_task];
  }

  // Tiles cut from one loop by the resource-aware pass share a `tile_group`.
  // They are threaded through the memref dependence-state SSA because that
  // version chain is linear -- taskflow has no fork/join over a partitioned
  // memref -- but the dimension they partition is dependence-free (the pass
  // only cuts a dimension that indexes every store, so the tiles touch disjoint
  // addresses). Reading the chain as a data dependence serialises the whole
  // group and reports an interval N times too large for an N-way cut.
  static int tileGroupOf(TaskflowTaskOp task) {
    if (auto attr = task->getAttrOfType<IntegerAttr>("tile_group"))
      return static_cast<int>(attr.getInt());
    return -1;
  }

  void buildTaskDependences() {
    successors_.resize(tasks_.size());
    predecessor_count_.assign(tasks_.size(), 0);
    start_times_.assign(tasks_.size(), 0);

    for (auto [task_idx, task] : llvm::enumerate(tasks_)) {
      const int task_tile_group = tileGroupOf(task);
      for (Value operand : task->getOperands()) {
        auto producer = operand.getDefiningOp<TaskflowTaskOp>();
        if (!producer) {
          continue;
        }
        // An ordered group (cut on a dimension that does not index every
        // store) keeps its chain; only a parallel-safe group drops it.
        auto tile_parallel_attr =
            task->getAttrOfType<BoolAttr>("tile_parallel");
        if (task_tile_group >= 0 && tileGroupOf(producer) == task_tile_group &&
            (!tile_parallel_attr || tile_parallel_attr.getValue())) {
          continue;
        }
        // Consuming a group means consuming ALL of it. TaskTiler rewires the
        // original consumers onto the last tile, so once the intra-group chain
        // is dropped the earlier tiles have no path to this task and could be
        // ordered after it. Depend on every member instead.
        const int producer_tile_group = tileGroupOf(producer);
        if (producer_tile_group >= 0 && producer_tile_group != task_tile_group) {
          for (auto [sibling_idx, sibling_task] : llvm::enumerate(tasks_)) {
            if (sibling_task == producer ||
                tileGroupOf(sibling_task) != producer_tile_group)
              continue;
            addExecutionOrderDependence(static_cast<int>(sibling_idx),
                                        static_cast<int>(task_idx));
            // Also record it as a real predecessor. This local graph decides
            // start times, but `TaskPipelineIntervalAnalyzer` builds its own
            // edge set from `predecessor_tasks` alone, and a sibling edge that
            // exists only here is invisible to the longest-path search that
            // produces the reported interval. The omission under-reports:
            // a consumer sharing a cell with tile 0 of an N-way group had no
            // path back to it, so that cell's cycle collapsed to bare
            // occupancy.
            schedule_result_[task_idx].predecessor_tasks.push_back(
                sibling_task);
            schedule_result_[sibling_idx].successor_tasks.push_back(task);
          }
        }

        auto producer_it = task_to_index_.find(producer.getOperation());
        if (producer_it == task_to_index_.end()) {
          continue;
        }

        int producer_idx = producer_it->second;
        addExecutionOrderDependence(producer_idx, static_cast<int>(task_idx));
        schedule_result_[task_idx].predecessor_tasks.push_back(producer);
        schedule_result_[producer_idx].successor_tasks.push_back(task);
      }
    }
  }

  void buildCgraExecutionOrderDependences() {
    DenseMap<int64_t, SmallVector<int>> cgra_location_to_tasks;
    for (auto [task_idx, task_result] : llvm::enumerate(schedule_result_)) {
      for (const TaskScheduleResult::CgraOccupancy &occupancy :
           task_result.cgra_occupancies) {
        cgra_location_to_tasks[encodeCgraLocation(occupancy)].push_back(
            static_cast<int>(task_idx));
      }
    }

    for (auto &entry : cgra_location_to_tasks) {
      SmallVector<int> &tasks = entry.second;
      llvm::sort(tasks, [&](int lhs, int rhs) {
        int lhs_context = getTaskContextOnCgra(lhs, entry.first);
        int rhs_context = getTaskContextOnCgra(rhs, entry.first);
        if (lhs_context != rhs_context) {
          return lhs_context < rhs_context;
        }
        return lhs < rhs;
      });

      for (size_t i = 1; i < tasks.size(); ++i) {
        addExecutionOrderDependence(tasks[i - 1], tasks[i]);
      }
    }
  }

  int getTaskContextOnCgra(int task_idx, int64_t cgra_location) const {
    for (const TaskScheduleResult::CgraOccupancy &occupancy :
         schedule_result_[task_idx].cgra_occupancies) {
      if (encodeCgraLocation(occupancy) == cgra_location) {
        return occupancy.context_id;
      }
    }
    return 0;
  }

  LogicalResult computeStartTimes() {
    SmallVector<int> ready_tasks;
    for (auto [task_idx, pred_count] : llvm::enumerate(predecessor_count_)) {
      if (pred_count == 0) {
        ready_tasks.push_back(static_cast<int>(task_idx));
      }
    }

    size_t processed_tasks = 0;
    while (!ready_tasks.empty()) {
      int task = ready_tasks.pop_back_val();
      ++processed_tasks;

      for (int next_task : successors_[task]) {
        int64_t next_start =
            start_times_[task] + schedule_result_[task].duration;
        start_times_[next_task] = std::max(start_times_[next_task], next_start);

        --predecessor_count_[next_task];
        if (predecessor_count_[next_task] == 0) {
          ready_tasks.push_back(next_task);
        }
      }
    }

    if (processed_tasks != tasks_.size()) {
      return func_.emitError()
             << "cannot analyze task pipeline interval because task "
                "dependences and CGRA context order form a cycle";
    }
    return success();
  }

  void updateScheduleTimes() {
    for (auto [task_idx, task_result] : llvm::enumerate(schedule_result_)) {
      int64_t start_time = start_times_[task_idx];
      task_result.start_time = start_time;
      task_result.end_time = start_time + task_result.duration;
      for (TaskScheduleResult::CgraOccupancy &occupancy :
           task_result.cgra_occupancies) {
        occupancy.start_time = start_time;
        occupancy.duration = task_result.duration;
      }
    }
  }

  func::FuncOp func_;
  SmallVector<TaskflowTaskOp> tasks_;
  DenseMap<Operation *, int> task_to_index_;
  DenseSet<int64_t> execution_order_edge_keys_;
  SmallVector<SmallVector<int>> successors_;
  SmallVector<int> predecessor_count_;
  SmallVector<int64_t> start_times_;
  SmallVector<TaskScheduleResult> schedule_result_;
};

static ArrayAttr buildTaskNameArray(MLIRContext *context,
                                    ArrayRef<TaskflowTaskOp> tasks) {
  SmallVector<Attribute> task_names;
  for (TaskflowTaskOp task : tasks) {
    task_names.push_back(StringAttr::get(context, task.getTaskName()));
  }
  return ArrayAttr::get(context, task_names);
}

static void emitPipelineIntervalInfo(func::FuncOp func,
                                     const TaskPipelineIntervalResult &result) {
  MLIRContext *context = func.getContext();
  OpBuilder builder(context);

  SmallVector<NamedAttribute> attrs;
  // `pipeline_interval` stays i32 because the in-tree expectations pin that
  // type. Whole-network kernels run past what i32 holds -- a GPT-2 prefill
  // block already measures 1.85e9 cycles -- so when the analysis exceeds it,
  // say so and publish the exact value alongside rather than emitting a
  // silently truncated number under the same name.
  const int64_t interval = result.pipeline_interval;
  attrs.push_back(builder.getNamedAttr(
      "pipeline_interval",
      builder.getI32IntegerAttr(
          static_cast<int32_t>(std::min<int64_t>(interval, INT32_MAX)))));
  if (interval > INT32_MAX) {
    func.emitWarning() << "pipeline interval " << interval
                       << " exceeds i32; `pipeline_interval` is clamped, see "
                          "`pipeline_interval_cycles` for the exact value";
    attrs.push_back(builder.getNamedAttr(
        "pipeline_interval_cycles", builder.getI64IntegerAttr(interval)));
  }

  StringRef bottleneck_task_name = "";
  if (result.bottleneck_task) {
    TaskflowTaskOp bottleneck_task = result.bottleneck_task;
    bottleneck_task_name = bottleneck_task.getTaskName();
  }
  attrs.push_back(builder.getNamedAttr(
      "bottleneck_task", builder.getStringAttr(bottleneck_task_name)));
  attrs.push_back(builder.getNamedAttr(
      "critical_path", buildTaskNameArray(context, result.critical_path)));

  func->setAttr("task_pipeline_interval_info",
                DictionaryAttr::get(context, attrs));
}

struct AnalyzeTaskPipelineIntervalPass
    : public PassWrapper<AnalyzeTaskPipelineIntervalPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnalyzeTaskPipelineIntervalPass)

  StringRef getArgument() const override {
    return "analyze-task-pipeline-interval";
  }

  StringRef getDescription() const override {
    return "Analyzes pipeline interval for profiled and scheduled Taskflow "
           "tasks";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    TaskScheduleResultParser parser(func);
    FailureOr<SmallVector<TaskScheduleResult>> schedule_result = parser.parse();
    if (failed(schedule_result)) {
      signalPassFailure();
      return;
    }

    TaskPipelineIntervalResult result =
        TaskPipelineIntervalAnalyzer(*schedule_result).analyze();
    emitPipelineIntervalInfo(func, result);
  }
};

} // namespace

namespace mlir {
namespace taskflow {

std::unique_ptr<Pass> createAnalyzeTaskPipelineIntervalPass() {
  return std::make_unique<AnalyzeTaskPipelineIntervalPass>();
}

} // namespace taskflow
} // namespace mlir
