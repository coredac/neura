// Hand-written schedule-analysis case for TaskPipelineIntervalAnalyzer.
//
// Data dependence: T0 -> T1 -> T2
// CGRA0: T0, then T2
// CGRA1: T1
//
// Expected analysis graph:
//   Data-dependence edges:     T0 -> T1, T1 -> T2
//   CGRA execution-order edge: T0 -> T2
//
// Expected pipeline cycles:
//   CGRA0: T0(this input) -> T1 -> T2 -> T0(next input)
//          interval = 2 + 3 + 5 = 10
//   CGRA1: T1(this input) -> T1(next input)
//          interval = 3
//
// Expected pipeline interval: 10


// RUN: mlir-neura-opt %s --analyze-task-pipeline-interval | FileCheck %s
//
// CHECK-LABEL: func.func @chain_reuse_cgra0
// CHECK-SAME: task_pipeline_interval_info = {bottleneck_task = "Task_2",
// CHECK-SAME: critical_path = ["Task_0", "Task_1", "Task_2"]
// CHECK-SAME: pipeline_interval = 10 : i32

module {
  func.func @chain_reuse_cgra0(%seed: i32) -> i32 {
    %t0 = taskflow.task @Task_0 value_inputs(%seed : i32)
        {profile_info = {duration = 2 : i32},
         task_orchestration_info = {cgra_positions = [{col = 0 : i32, context_id = 0 : i32, row = 0 : i32}],
                                    read_sram_locations = [],
                                    write_sram_locations = []}}
        : (i32) -> (i32) {
    ^bb0(%arg0: i32):
      taskflow.yield values(%arg0 : i32)
    }

    %t1 = taskflow.task @Task_1 value_inputs(%t0 : i32)
        {profile_info = {duration = 3 : i32},
         task_orchestration_info = {cgra_positions = [{col = 1 : i32, context_id = 0 : i32, row = 0 : i32}],
                                    read_sram_locations = [],
                                    write_sram_locations = []}}
        : (i32) -> (i32) {
    ^bb0(%arg0: i32):
      taskflow.yield values(%arg0 : i32)
    }

    %t2 = taskflow.task @Task_2 value_inputs(%t1 : i32)
        {profile_info = {duration = 5 : i32},
         task_orchestration_info = {cgra_positions = [{col = 0 : i32, context_id = 1 : i32, row = 0 : i32}],
                                    read_sram_locations = [],
                                    write_sram_locations = []}}
        : (i32) -> (i32) {
    ^bb0(%arg0: i32):
      taskflow.yield values(%arg0 : i32)
    }

    return %t2 : i32
  }
}
