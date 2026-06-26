// Tests that orchestrate-task-on-cgra correctly handles pre-annotated
// resource-binding attributes (cgra_count, cgra_shape) and produces
// task_mapping_info while removing the consumed attributes.
//
// The input is a task graph where:
//   Task_0 (cgra_count=2, cgra_shape="1x2") writes %A, read by Task_1.
//   Task_1 (cgra_count=1) reads %A, writes %B — SSA-dependent on Task_0.
//   Task_2 (cgra_count=1) reads %B — SSA-dependent on Task_1.
// The three tasks form a linear chain: Task_0 → Task_1 → Task_2.

// RUN: mlir-neura-opt %s --orchestrate-task-on-cgra \
// RUN:   --architecture-spec=%S/../../../arch_spec/architecture_4x4.yaml \
// RUN:   -o %t.allocated_4x4.mlir
// RUN: FileCheck %s --input-file=%t.allocated_4x4.mlir

module {
  func.func @resource_binding_chain(
      %A: memref<64xf32>, %B: memref<64xf32>, %C: memref<64xf32>,
      %val: f32) {

    // Task_0: writes %A.  Allocated to 2 CGRAs with shape 1x2.
    %dr0, %dw0 = taskflow.task @Task_0
        will_read(%A : memref<64xf32>)
        will_write(%A : memref<64xf32>)
        value_inputs(%val : f32)
        [original_read_memrefs(%A : memref<64xf32>),
         original_write_memrefs(%A : memref<64xf32>)]
        {cgra_count = 2 : i32, cgra_shape = "1x2"}
        : (memref<64xf32>, memref<64xf32>, f32)
       -> (memref<64xf32>, memref<64xf32>) {
    ^bb0(%a0: memref<64xf32>, %a1: memref<64xf32>, %v: f32):
      taskflow.yield done_read(%a0 : memref<64xf32>) done_write(%a1 : memref<64xf32>)
    }

    // Task_1: reads %A (via Task_0 output), writes %B.  Single CGRA.
    %dr1, %dw1 = taskflow.task @Task_1
        will_read(%dw0 : memref<64xf32>)
        will_write(%B : memref<64xf32>)
        value_inputs(%dr0 : memref<64xf32>)
        [original_read_memrefs(%A : memref<64xf32>),
         original_write_memrefs(%B : memref<64xf32>)]
        {cgra_count = 1 : i32}
        : (memref<64xf32>, memref<64xf32>, memref<64xf32>)
       -> (memref<64xf32>, memref<64xf32>) {
    ^bb0(%a0: memref<64xf32>, %b0: memref<64xf32>, %a1: memref<64xf32>):
      taskflow.yield done_read(%a0 : memref<64xf32>) done_write(%b0 : memref<64xf32>)
    }

    // Task_2: reads %B (via Task_1 output), writes %C.  Single CGRA.
    %dr2, %dw2 = taskflow.task @Task_2
        will_read(%dw1 : memref<64xf32>)
        will_write(%C : memref<64xf32>)
        value_inputs(%dr1 : memref<64xf32>)
        [original_read_memrefs(%B : memref<64xf32>),
         original_write_memrefs(%C : memref<64xf32>)]
        {cgra_count = 1 : i32}
        : (memref<64xf32>, memref<64xf32>, memref<64xf32>)
       -> (memref<64xf32>, memref<64xf32>) {
    ^bb0(%b0: memref<64xf32>, %c0: memref<64xf32>, %b1: memref<64xf32>):
      taskflow.yield done_read(%b0 : memref<64xf32>) done_write(%c0 : memref<64xf32>)
    }

    return
  }
}

// CHECK: module {
// CHECK-NEXT:   func.func @resource_binding_chain(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>, %arg3: f32) {
// CHECK-NEXT:     %done_read, %done_write = taskflow.task @Task_0 will_read(%arg0 : memref<64xf32>) will_write(%arg0 : memref<64xf32>) value_inputs(%arg3 : f32) [original_read_memrefs(%arg0 : memref<64xf32>), original_write_memrefs(%arg0 : memref<64xf32>)] {profile_info = {duration = 1 : i32}, task_orchestration_info = {cgra_positions = [{col = 0 : i32, context_id = 0 : i32, row = 0 : i32}, {col = 1 : i32, context_id = 0 : i32, row = 0 : i32}], read_sram_locations = [{col = 1 : i32, row = 0 : i32}], write_sram_locations = [{col = 1 : i32, row = 0 : i32}]}} : (memref<64xf32>, memref<64xf32>, f32) -> (memref<64xf32>, memref<64xf32>) {
// CHECK-NEXT:     ^bb0(%arg4: memref<64xf32>, %arg5: memref<64xf32>, %arg6: f32):
// CHECK-NEXT:       taskflow.yield done_read(%arg4 : memref<64xf32>) done_write(%arg5 : memref<64xf32>)
// CHECK-NEXT:     }
// CHECK-NEXT:     %done_read_0, %done_write_1 = taskflow.task @Task_1 will_read(%done_write : memref<64xf32>) will_write(%arg1 : memref<64xf32>) value_inputs(%done_read : memref<64xf32>) [original_read_memrefs(%arg0 : memref<64xf32>), original_write_memrefs(%arg1 : memref<64xf32>)] {profile_info = {duration = 1 : i32}, task_orchestration_info = {cgra_positions = [{col = 2 : i32, context_id = 0 : i32, row = 0 : i32}], read_sram_locations = [{col = 1 : i32, row = 0 : i32}], write_sram_locations = [{col = 3 : i32, row = 0 : i32}]}} : (memref<64xf32>, memref<64xf32>, memref<64xf32>) -> (memref<64xf32>, memref<64xf32>) {
// CHECK-NEXT:     ^bb0(%arg4: memref<64xf32>, %arg5: memref<64xf32>, %arg6: memref<64xf32>):
// CHECK-NEXT:       taskflow.yield done_read(%arg4 : memref<64xf32>) done_write(%arg5 : memref<64xf32>)
// CHECK-NEXT:     }
// CHECK-NEXT:     %done_read_2, %done_write_3 = taskflow.task @Task_2 will_read(%done_write_1 : memref<64xf32>) will_write(%arg2 : memref<64xf32>) value_inputs(%done_read_0 : memref<64xf32>) [original_read_memrefs(%arg1 : memref<64xf32>), original_write_memrefs(%arg2 : memref<64xf32>)] {profile_info = {duration = 1 : i32}, task_orchestration_info = {cgra_positions = [{col = 3 : i32, context_id = 0 : i32, row = 0 : i32}], read_sram_locations = [{col = 3 : i32, row = 0 : i32}], write_sram_locations = [{col = 3 : i32, row = 0 : i32}]}} : (memref<64xf32>, memref<64xf32>, memref<64xf32>) -> (memref<64xf32>, memref<64xf32>) {
// CHECK-NEXT:     ^bb0(%arg4: memref<64xf32>, %arg5: memref<64xf32>, %arg6: memref<64xf32>):
// CHECK-NEXT:       taskflow.yield done_read(%arg4 : memref<64xf32>) done_write(%arg5 : memref<64xf32>)
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT: }
