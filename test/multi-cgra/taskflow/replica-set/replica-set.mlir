// Every replica of a data-parallel task must be placed, or the interval is a
// measurement of work no hardware performs.
//
// Four independent tasks, each asking for a 1x2 tile array replicated twice:
// 4 tasks x 2 cgras x 2 replicas = 16 cells, exactly the 4x4 grid, all at one
// instant. There is an exact packing and the orchestrator has to find it.
//
// It did not. Placing the first replica where it scored best and pinning the
// rest into what was left let a task fragment the grid for itself: three tasks
// held 12 cells as six scattered 1x2 arrays, and the fourth found its two free
// cells non-adjacent, dropped a replica, and warned that est_latency still
// assumed two. `replicas=1/2` in the schedule below is that bug.
//
// `est_latency` is stated rather than derived so the test pins the placer and
// not the cost model.

// RUN: mlir-neura-opt %s \
// RUN:   --orchestrate-tasks-on-accelerators=scheduling-mode=spatial-temporal \
// RUN:   --architecture-spec=%S/../../../arch_spec/architecture_4x4.yaml \
// RUN:   -o /dev/null 2>&1 | FileCheck %s

// Every task holds its whole set: 2 cgras x 2 replicas = 4 cells.
// CHECK-COUNT-4: cgras=4 replicas=2/2
// CHECK: grid_utilisation=1.000

module {
  func.func @replica_set(%A: memref<64xf32>, %B: memref<64xf32>,
                         %C: memref<64xf32>, %D: memref<64xf32>,
                         %val: f32) {
    %dr0, %dw0 = taskflow.task @Task_0
        will_reads(%A : memref<64xf32>)
        will_writes(%A : memref<64xf32>)
        value_inputs(%val : f32)
        [original_read_memrefs(%A : memref<64xf32>),
         original_write_memrefs(%A : memref<64xf32>)]
        {cgra_count = 2 : i32, cgra_shape = "1x2", replicas = 2 : i32,
         est_latency = 260 : i64}
        : (memref<64xf32>, memref<64xf32>, f32)
       -> (memref<64xf32>, memref<64xf32>) {
    ^bb0(%a0: memref<64xf32>, %a1: memref<64xf32>, %v: f32):
      taskflow.yield done_reads(%a0 : memref<64xf32>) done_writes(%a1 : memref<64xf32>)
    }

    %dr1, %dw1 = taskflow.task @Task_1
        will_reads(%B : memref<64xf32>)
        will_writes(%B : memref<64xf32>)
        value_inputs(%val : f32)
        [original_read_memrefs(%B : memref<64xf32>),
         original_write_memrefs(%B : memref<64xf32>)]
        {cgra_count = 2 : i32, cgra_shape = "1x2", replicas = 2 : i32,
         est_latency = 260 : i64}
        : (memref<64xf32>, memref<64xf32>, f32)
       -> (memref<64xf32>, memref<64xf32>) {
    ^bb0(%b0: memref<64xf32>, %b1: memref<64xf32>, %v: f32):
      taskflow.yield done_reads(%b0 : memref<64xf32>) done_writes(%b1 : memref<64xf32>)
    }

    %dr2, %dw2 = taskflow.task @Task_2
        will_reads(%C : memref<64xf32>)
        will_writes(%C : memref<64xf32>)
        value_inputs(%val : f32)
        [original_read_memrefs(%C : memref<64xf32>),
         original_write_memrefs(%C : memref<64xf32>)]
        {cgra_count = 2 : i32, cgra_shape = "1x2", replicas = 2 : i32,
         est_latency = 260 : i64}
        : (memref<64xf32>, memref<64xf32>, f32)
       -> (memref<64xf32>, memref<64xf32>) {
    ^bb0(%c0: memref<64xf32>, %c1: memref<64xf32>, %v: f32):
      taskflow.yield done_reads(%c0 : memref<64xf32>) done_writes(%c1 : memref<64xf32>)
    }

    %dr3, %dw3 = taskflow.task @Task_3
        will_reads(%D : memref<64xf32>)
        will_writes(%D : memref<64xf32>)
        value_inputs(%val : f32)
        [original_read_memrefs(%D : memref<64xf32>),
         original_write_memrefs(%D : memref<64xf32>)]
        {cgra_count = 2 : i32, cgra_shape = "1x2", replicas = 2 : i32,
         est_latency = 260 : i64}
        : (memref<64xf32>, memref<64xf32>, f32)
       -> (memref<64xf32>, memref<64xf32>) {
    ^bb0(%d0: memref<64xf32>, %d1: memref<64xf32>, %v: f32):
      taskflow.yield done_reads(%d0 : memref<64xf32>) done_writes(%d1 : memref<64xf32>)
    }

    return
  }
}
