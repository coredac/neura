// Tests that orchestrate-tasks-on-accelerators correctly handles pre-annotated
// resource-binding attributes (cgra_count, cgra_shape) and produces
// task_mapping_info while removing the consumed attributes.
//
// The input is a task graph where:
//   Task_0 (cgra_count=2, cgra_shape="1x2") writes %A, read by Task_1.
//   Task_1 (cgra_count=1) reads %A, writes %B — SSA-dependent on Task_0.
//   Task_2 (cgra_count=1) reads %B — SSA-dependent on Task_1.
// The three tasks form a linear chain: Task_0 → Task_1 → Task_2.

// RUN: mlir-neura-opt %s --orchestrate-tasks-on-accelerators \
// RUN:   --architecture-spec=%S/../../../arch_spec/architecture_4x4.yaml \
// RUN:   -o %t.allocated_4x4.mlir
// RUN: FileCheck %s --input-file=%t.allocated_4x4.mlir

module {
  func.func @resource_binding_chain(
      %A: memref<64xf32>, %B: memref<64xf32>, %C: memref<64xf32>,
      %val: f32) {

    // Task_0: writes %A.  Allocated to 2 CGRAs with shape 1x2.
    %dr0, %dw0 = taskflow.task @Task_0
        will_reads(%A : memref<64xf32>)
        will_writes(%A : memref<64xf32>)
        value_inputs(%val : f32)
        [original_read_memrefs(%A : memref<64xf32>),
         original_write_memrefs(%A : memref<64xf32>)]
        {cgra_count = 2 : i32, cgra_shape = "1x2"}
        : (memref<64xf32>, memref<64xf32>, f32)
       -> (memref<64xf32>, memref<64xf32>) {
    ^bb0(%a0: memref<64xf32>, %a1: memref<64xf32>, %v: f32):
      taskflow.yield done_reads(%a0 : memref<64xf32>) done_writes(%a1 : memref<64xf32>)
    }

    // Task_1: reads %A (via Task_0 output), writes %B.  Single CGRA.
    %dr1, %dw1 = taskflow.task @Task_1
        will_reads(%dw0 : memref<64xf32>)
        will_writes(%B : memref<64xf32>)
        value_inputs(%dr0 : memref<64xf32>)
        [original_read_memrefs(%A : memref<64xf32>),
         original_write_memrefs(%B : memref<64xf32>)]
        {cgra_count = 1 : i32}
        : (memref<64xf32>, memref<64xf32>, memref<64xf32>)
       -> (memref<64xf32>, memref<64xf32>) {
    ^bb0(%a0: memref<64xf32>, %b0: memref<64xf32>, %a1: memref<64xf32>):
      taskflow.yield done_reads(%a0 : memref<64xf32>) done_writes(%b0 : memref<64xf32>)
    }

    // Task_2: reads %B (via Task_1 output), writes %C.  Single CGRA.
    %dr2, %dw2 = taskflow.task @Task_2
        will_reads(%dw1 : memref<64xf32>)
        will_writes(%C : memref<64xf32>)
        value_inputs(%dr1 : memref<64xf32>)
        [original_read_memrefs(%B : memref<64xf32>),
         original_write_memrefs(%C : memref<64xf32>)]
        {cgra_count = 1 : i32}
        : (memref<64xf32>, memref<64xf32>, memref<64xf32>)
       -> (memref<64xf32>, memref<64xf32>) {
    ^bb0(%b0: memref<64xf32>, %c0: memref<64xf32>, %b1: memref<64xf32>):
      taskflow.yield done_reads(%b0 : memref<64xf32>) done_writes(%c0 : memref<64xf32>)
    }

    return
  }
}

// CHECK: module {
// CHECK-NEXT:   func.func @resource_binding_chain(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>, %arg3: f32) {
// CHECK-NEXT:     %done_reads, %done_writes = taskflow.task @Task_0 will_reads(%arg0 : memref<64xf32>) will_writes(%arg0 : memref<64xf32>) value_inputs(%arg3 : f32) [original_read_memrefs(%arg0 : memref<64xf32>), original_write_memrefs(%arg0 : memref<64xf32>)] {profile_info = {duration = 1 : i32}, task_orchestration_info = {cgra_positions = [{col = 0 : i32, context_id = 0 : i32, row = 0 : i32}, {col = 1 : i32, context_id = 0 : i32, row = 0 : i32}], read_sram_locations = [{col = 1 : i32, row = 0 : i32}], write_sram_locations = [{col = 1 : i32, row = 0 : i32}]}} : (memref<64xf32>, memref<64xf32>, f32) -> (memref<64xf32>, memref<64xf32>) {
// CHECK-NEXT:     ^bb0(%arg4: memref<64xf32>, %arg5: memref<64xf32>, %arg6: f32):
// CHECK-NEXT:       taskflow.yield done_reads(%arg4 : memref<64xf32>) done_writes(%arg5 : memref<64xf32>)
// CHECK-NEXT:     }
// CHECK-NEXT:     %done_reads_0, %done_writes_1 = taskflow.task @Task_1 will_reads(%done_writes : memref<64xf32>) will_writes(%arg1 : memref<64xf32>) value_inputs(%done_reads : memref<64xf32>) [original_read_memrefs(%arg0 : memref<64xf32>), original_write_memrefs(%arg1 : memref<64xf32>)] {profile_info = {duration = 1 : i32}, task_orchestration_info = {cgra_positions = [{col = 2 : i32, context_id = 0 : i32, row = 0 : i32}], read_sram_locations = [{col = 1 : i32, row = 0 : i32}], write_sram_locations = [{col = 3 : i32, row = 0 : i32}]}} : (memref<64xf32>, memref<64xf32>, memref<64xf32>) -> (memref<64xf32>, memref<64xf32>) {
// CHECK-NEXT:     ^bb0(%arg4: memref<64xf32>, %arg5: memref<64xf32>, %arg6: memref<64xf32>):
// CHECK-NEXT:       taskflow.yield done_reads(%arg4 : memref<64xf32>) done_writes(%arg5 : memref<64xf32>)
// CHECK-NEXT:     }
// CHECK-NEXT:     %done_reads_2, %done_writes_3 = taskflow.task @Task_2 will_reads(%done_writes_1 : memref<64xf32>) will_writes(%arg2 : memref<64xf32>) value_inputs(%done_reads_0 : memref<64xf32>) [original_read_memrefs(%arg1 : memref<64xf32>), original_write_memrefs(%arg2 : memref<64xf32>)] {profile_info = {duration = 1 : i32}, task_orchestration_info = {cgra_positions = [{col = 3 : i32, context_id = 0 : i32, row = 0 : i32}], read_sram_locations = [{col = 3 : i32, row = 0 : i32}], write_sram_locations = [{col = 3 : i32, row = 0 : i32}]}} : (memref<64xf32>, memref<64xf32>, memref<64xf32>) -> (memref<64xf32>, memref<64xf32>) {
// CHECK-NEXT:     ^bb0(%arg4: memref<64xf32>, %arg5: memref<64xf32>, %arg6: memref<64xf32>):
// CHECK-NEXT:       taskflow.yield done_reads(%arg4 : memref<64xf32>) done_writes(%arg5 : memref<64xf32>)
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT: }
