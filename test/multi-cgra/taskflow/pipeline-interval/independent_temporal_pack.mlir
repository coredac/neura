// Hand-written schedule-analysis case for TaskPipelineIntervalAnalyzer.
//
// Data dependence: none
// CGRA0: T0, then T1
//
// Expected analysis graph:
//   Data-dependence edges:     none
//   CGRA execution-order edge: T0 -> T1
//
// Expected pipeline cycle:
//   CGRA0: T0(this input) -> T1 -> T0(next input)
//          interval = 7 + 2 = 9
//
// Expected pipeline interval: 9

// RUN: mlir-neura-opt %s --analyze-task-pipeline-interval | FileCheck %s
//
// CHECK-LABEL: func.func @independent_temporal_pack
// CHECK-SAME: task_pipeline_interval_info = {bottleneck_task = "Task_0",
// CHECK-SAME: critical_path = ["Task_0", "Task_1"]
// CHECK-SAME: pipeline_interval = 9 : i32

module {
  func.func @independent_temporal_pack(%seed0: i32, %seed1: i32) -> (i32, i32) {
    %t0 = taskflow.task @Task_0 value_inputs(%seed0 : i32)
        {profile_info = {duration = 7 : i32},
         task_orchestration_info = {cgra_positions = [{col = 0 : i32, context_id = 0 : i32, row = 0 : i32}],
                                    read_sram_locations = [],
                                    write_sram_locations = []}}
        : (i32) -> (i32) {
    ^bb0(%arg0: i32):
      taskflow.yield values(%arg0 : i32)
    }

    %t1 = taskflow.task @Task_1 value_inputs(%seed1 : i32)
        {profile_info = {duration = 2 : i32},
         task_orchestration_info = {cgra_positions = [{col = 0 : i32, context_id = 1 : i32, row = 0 : i32}],
                                    read_sram_locations = [],
                                    write_sram_locations = []}}
        : (i32) -> (i32) {
    ^bb0(%arg0: i32):
      taskflow.yield values(%arg0 : i32)
    }

    return %t0, %t1 : i32, i32
  }
}
