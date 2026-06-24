// RUN: mlir-neura-opt %s --affine-loop-tree-serialization \
// RUN: -o %t.serialized.mlir
// RUN: FileCheck %s --input-file=%t.serialized.mlir --check-prefixes=SERIALIZED

// RUN: mlir-neura-opt %s --affine-loop-tree-serialization \
// RUN: --affine-loop-perfection \
// RUN: -o %t.perfect.mlir
// RUN: FileCheck %s --input-file=%t.perfect.mlir --check-prefixes=PERFECT

// RUN: mlir-neura-opt %s --affine-loop-tree-serialization \
// RUN: --convert-affine-to-taskflow \
// RUN: -o %t.taskflow.mlir
// RUN: FileCheck %s --input-file=%t.taskflow.mlir --check-prefixes=TASKFLOW

// RUN: mlir-neura-opt %s --affine-loop-tree-serialization \
// RUN: --convert-affine-to-taskflow \
// RUN: --memory-access-streaming-fusion \
// RUN: -o %t.stream.mlir
// RUN: FileCheck %s --input-file=%t.stream.mlir --check-prefixes=STREAM

// RUN: mlir-neura-opt %t.stream.mlir \
// RUN: --affine-loop-tree-serialization \
// RUN: --affine-loop-perfection \
// RUN: --construct-hyperblock-from-task \
// RUN: --classify-task-and-counter \
// RUN: --convert-taskflow-to-neura \
// RUN: --lower-affine \
// RUN: --convert-scf-to-cf \
// RUN: --convert-cf-to-llvm \
// RUN: --assign-accelerator \
// RUN: --lower-memref-to-neura \
// RUN: --lower-arith-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: --promote-input-arg-to-const \
// RUN: --fold-constant \
// RUN: --canonicalize-return \
// RUN: --canonicalize-live-in \
// RUN: --leverage-predicated-value \
// RUN: --transform-ctrl-to-data-flow \
// RUN: --fold-constant \
// RUN: '--resource-aware-task-optimization=balance-skip-mapper=false' \
// RUN: --architecture-spec=%S/../../../arch_spec/architecture_with_counter.yaml \
// RUN: -o %t.resopt.mlir
// RUN: FileCheck %s --input-file=%t.resopt.mlir --check-prefixes=RESOPT

// RUN: mlir-neura-opt %s --affine-loop-tree-serialization \
// RUN: --convert-affine-to-taskflow \
// RUN: --construct-hyperblock-from-task \
// RUN: --convert-taskflow-to-neura \
// RUN: --architecture-spec=%S/../../../arch_spec/architecture.yaml \
// RUN: -o %t.kernel.mlir
// RUN: FileCheck %s --input-file=%t.kernel.mlir --check-prefixes=KERNEL

// RUN: mlir-neura-opt %s --affine-loop-tree-serialization \
// RUN: --convert-affine-to-taskflow \
// RUN: --construct-hyperblock-from-task \
// RUN: -o %t.hyperblock.mlir
// RUN: FileCheck %s --input-file=%t.hyperblock.mlir --check-prefixes=HYPERBLOCK


// RUN: mlir-neura-opt %s --affine-loop-tree-serialization \
// RUN: --convert-affine-to-taskflow \
// RUN: --construct-hyperblock-from-task \
// RUN: '--orchestrate-task-on-cgra=scheduling-mode=spatial-temporal' \
// RUN: --architecture-spec=%S/../../../arch_spec/architecture_4x4.yaml \
// RUN: -o %t.map_4x4_spatial_temporal.mlir
// RUN: FileCheck %s --input-file=%t.map_4x4_spatial_temporal.mlir --check-prefixes=MAP-SPATIAL-TEMPORAL-4x4

// SMALL-GRID-1x1: Tests spatial-temporal mapping on a 1x1 CGRA grid.
// SMALL-GRID-1x1: 5 tasks must time-multiplex on the single CGRA.
// RUN: mlir-neura-opt %s --affine-loop-tree-serialization \
// RUN: --convert-affine-to-taskflow \
// RUN: --construct-hyperblock-from-task \
// RUN: '--orchestrate-task-on-cgra=scheduling-mode=spatial-temporal' \
// RUN: --architecture-spec=%S/../../../arch_spec/architecture_with_counter.yaml \
// RUN: -o %t.map_1x1_spatial_temporal.mlir
// RUN: FileCheck %s --input-file=%t.map_1x1_spatial_temporal.mlir --check-prefixes=MAP-SPATIAL-TEMPORAL-1x1

// SMALL-GRID-1x2: Tests spatial-temporal mapping on a 1x2 CGRA grid.
// SMALL-GRID-1x2: 2 CGRAs for 5 tasks — multiple temporal reuse slots.
// RUN: mlir-neura-opt %s --affine-loop-tree-serialization \
// RUN: --convert-affine-to-taskflow \
// RUN: --construct-hyperblock-from-task \
// RUN: '--orchestrate-task-on-cgra=scheduling-mode=spatial-temporal' \
// RUN: --architecture-spec=%S/../../../arch_spec/architecture_1x2.yaml \
// RUN: -o %t.map_1x2_spatial_temporal.mlir
// RUN: FileCheck %s --input-file=%t.map_1x2_spatial_temporal.mlir --check-prefixes=MAP-SPATIAL-TEMPORAL-1x2

// RUN: mlir-neura-opt %s --affine-loop-tree-serialization \
// RUN: --convert-affine-to-taskflow \
// RUN: --construct-hyperblock-from-task \
// RUN: '--orchestrate-task-on-cgra=scheduling-mode=spatial' \
// RUN: --architecture-spec=%S/../../../arch_spec/architecture_4x4.yaml \
// RUN: -o %t.map_4x4_spatial.mlir
// RUN: FileCheck %s --input-file=%t.map_4x4_spatial.mlir --check-prefixes=MAP-SPATIAL

module attributes {} {
  func.func @_Z21pureNestedLoopExamplePA8_A6_iPA8_A5_iS4_PA7_iPA9_iPiS9_S9_S9_S9_(%arg0: memref<?x8x6xi32>, %arg1: memref<?x8x5xi32>, %arg2: memref<?x8x5xi32>, %arg3: memref<?x7xi32>, %arg4: memref<?x9xi32>, %arg5: memref<?xi32>, %arg6: memref<?xi32>, %arg7: memref<?xi32>, %arg8: memref<?xi32>, %arg9: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg10 = 0 to 4 {
      affine.for %arg11 = 0 to 8 {
        affine.for %arg12 = 0 to 6 {
          %1 = affine.load %arg0[%arg10, %arg11, %arg12] : memref<?x8x6xi32>
          affine.store %1, %arg5[%arg12] : memref<?xi32>
        }
        affine.for %arg12 = 0 to 5 {
          %1 = affine.load %arg1[%arg10, %arg11, %arg12] : memref<?x8x5xi32>
          %2 = affine.load %arg2[%arg10, %arg11, %arg12] : memref<?x8x5xi32>
          %3 = arith.addi %1, %2 : i32
          affine.store %3, %arg6[%arg12] : memref<?xi32>
        }
        affine.for %arg12 = 0 to 6 {
          %1 = affine.load %arg5[%arg12] : memref<?xi32>
          %2 = affine.load %arg6[%arg12] : memref<?xi32>
          %3 = arith.addi %1, %2 : i32
          %4 = affine.load %arg9[0] : memref<?xi32>
          %5 = arith.addi %4, %3 : i32
          affine.store %5, %arg9[0] : memref<?xi32>
        }
      }
      affine.for %arg11 = 0 to 7 {
        %1 = affine.load %arg3[%arg10, %arg11] : memref<?x7xi32>
        affine.store %1, %arg7[%arg11] : memref<?xi32>
      }
      affine.for %arg11 = 0 to 9 {
        %1 = affine.load %arg4[%arg10, %arg11] : memref<?x9xi32>
        %2 = affine.load %arg7[%arg11] : memref<?xi32>
        %3 = arith.addi %1, %2 : i32
        affine.store %3, %arg8[%arg11] : memref<?xi32>
      }
    }
    %0 = affine.load %arg9[0] : memref<?xi32>
    return %0 : i32
  }
}

// SERIALIZED: module {
// SERIALIZED-NEXT:   func.func @_Z21pureNestedLoopExamplePA8_A6_iPA8_A5_iS4_PA7_iPA9_iPiS9_S9_S9_S9_(%arg0: memref<?x8x6xi32>, %arg1: memref<?x8x5xi32>, %arg2: memref<?x8x5xi32>, %arg3: memref<?x7xi32>, %arg4: memref<?x9xi32>, %arg5: memref<?xi32>, %arg6: memref<?xi32>, %arg7: memref<?xi32>, %arg8: memref<?xi32>, %arg9: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// SERIALIZED-NEXT:     affine.for %arg10 = 0 to 4 {
// SERIALIZED-NEXT:       affine.for %arg11 = 0 to 8 {
// SERIALIZED-NEXT:         affine.for %arg12 = 0 to 6 {
// SERIALIZED-NEXT:           %1 = affine.load %arg0[%arg10, %arg11, %arg12] : memref<?x8x6xi32>
// SERIALIZED-NEXT:           affine.store %1, %arg5[%arg12] : memref<?xi32>
// SERIALIZED-NEXT:         }
// SERIALIZED-NEXT:       }
// SERIALIZED-NEXT:     }
// SERIALIZED-NEXT:     affine.for %arg10 = 0 to 4 {
// SERIALIZED-NEXT:       affine.for %arg11 = 0 to 8 {
// SERIALIZED-NEXT:         affine.for %arg12 = 0 to 5 {
// SERIALIZED-NEXT:           %1 = affine.load %arg1[%arg10, %arg11, %arg12] : memref<?x8x5xi32>
// SERIALIZED-NEXT:           %2 = affine.load %arg2[%arg10, %arg11, %arg12] : memref<?x8x5xi32>
// SERIALIZED-NEXT:           %3 = arith.addi %1, %2 : i32
// SERIALIZED-NEXT:           affine.store %3, %arg6[%arg12] : memref<?xi32>
// SERIALIZED-NEXT:         }
// SERIALIZED-NEXT:       }
// SERIALIZED-NEXT:     }
// SERIALIZED-NEXT:     affine.for %arg10 = 0 to 4 {
// SERIALIZED-NEXT:       affine.for %arg11 = 0 to 8 {
// SERIALIZED-NEXT:         affine.for %arg12 = 0 to 6 {
// SERIALIZED-NEXT:           %1 = affine.load %arg5[%arg12] : memref<?xi32>
// SERIALIZED-NEXT:           %2 = affine.load %arg6[%arg12] : memref<?xi32>
// SERIALIZED-NEXT:           %3 = arith.addi %1, %2 : i32
// SERIALIZED-NEXT:           %4 = affine.load %arg9[0] : memref<?xi32>
// SERIALIZED-NEXT:           %5 = arith.addi %4, %3 : i32
// SERIALIZED-NEXT:           affine.store %5, %arg9[0] : memref<?xi32>
// SERIALIZED-NEXT:         }
// SERIALIZED-NEXT:       }
// SERIALIZED-NEXT:     }
// SERIALIZED-NEXT:     affine.for %arg10 = 0 to 4 {
// SERIALIZED-NEXT:       affine.for %arg11 = 0 to 7 {
// SERIALIZED-NEXT:         %1 = affine.load %arg3[%arg10, %arg11] : memref<?x7xi32>
// SERIALIZED-NEXT:         affine.store %1, %arg7[%arg11] : memref<?xi32>
// SERIALIZED-NEXT:       }
// SERIALIZED-NEXT:     }
// SERIALIZED-NEXT:     affine.for %arg10 = 0 to 4 {
// SERIALIZED-NEXT:       affine.for %arg11 = 0 to 9 {
// SERIALIZED-NEXT:         %1 = affine.load %arg4[%arg10, %arg11] : memref<?x9xi32>
// SERIALIZED-NEXT:         %2 = affine.load %arg7[%arg11] : memref<?xi32>
// SERIALIZED-NEXT:         %3 = arith.addi %1, %2 : i32
// SERIALIZED-NEXT:         affine.store %3, %arg8[%arg11] : memref<?xi32>
// SERIALIZED-NEXT:       }
// SERIALIZED-NEXT:     }
// SERIALIZED-NEXT:     %0 = affine.load %arg9[0] : memref<?xi32>
// SERIALIZED-NEXT:     return %0 : i32
// SERIALIZED-NEXT:   }
// SERIALIZED-NEXT: }

// PERFECT: module {
// PERFECT-NEXT:   func.func @_Z21pureNestedLoopExamplePA8_A6_iPA8_A5_iS4_PA7_iPA9_iPiS9_S9_S9_S9_(%arg0: memref<?x8x6xi32>, %arg1: memref<?x8x5xi32>, %arg2: memref<?x8x5xi32>, %arg3: memref<?x7xi32>, %arg4: memref<?x9xi32>, %arg5: memref<?xi32>, %arg6: memref<?xi32>, %arg7: memref<?xi32>, %arg8: memref<?xi32>, %arg9: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// PERFECT-NEXT:     affine.for %arg10 = 0 to 4 {
// PERFECT-NEXT:       affine.for %arg11 = 0 to 8 {
// PERFECT-NEXT:         affine.for %arg12 = 0 to 6 {
// PERFECT-NEXT:           %1 = affine.load %arg0[%arg10, %arg11, %arg12] : memref<?x8x6xi32>
// PERFECT-NEXT:           affine.store %1, %arg5[%arg12] : memref<?xi32>
// PERFECT-NEXT:         }
// PERFECT-NEXT:       }
// PERFECT-NEXT:     }
// PERFECT-NEXT:     affine.for %arg10 = 0 to 4 {
// PERFECT-NEXT:       affine.for %arg11 = 0 to 8 {
// PERFECT-NEXT:         affine.for %arg12 = 0 to 5 {
// PERFECT-NEXT:           %1 = affine.load %arg1[%arg10, %arg11, %arg12] : memref<?x8x5xi32>
// PERFECT-NEXT:           %2 = affine.load %arg2[%arg10, %arg11, %arg12] : memref<?x8x5xi32>
// PERFECT-NEXT:           %3 = arith.addi %1, %2 : i32
// PERFECT-NEXT:           affine.store %3, %arg6[%arg12] : memref<?xi32>
// PERFECT-NEXT:         }
// PERFECT-NEXT:       }
// PERFECT-NEXT:     }
// PERFECT-NEXT:     affine.for %arg10 = 0 to 4 {
// PERFECT-NEXT:       affine.for %arg11 = 0 to 8 {
// PERFECT-NEXT:         affine.for %arg12 = 0 to 6 {
// PERFECT-NEXT:           %1 = affine.load %arg5[%arg12] : memref<?xi32>
// PERFECT-NEXT:           %2 = affine.load %arg6[%arg12] : memref<?xi32>
// PERFECT-NEXT:           %3 = arith.addi %1, %2 : i32
// PERFECT-NEXT:           %4 = affine.load %arg9[0] : memref<?xi32>
// PERFECT-NEXT:           %5 = arith.addi %4, %3 : i32
// PERFECT-NEXT:           affine.store %5, %arg9[0] : memref<?xi32>
// PERFECT-NEXT:         }
// PERFECT-NEXT:       }
// PERFECT-NEXT:     }
// PERFECT-NEXT:     affine.for %arg10 = 0 to 4 {
// PERFECT-NEXT:       affine.for %arg11 = 0 to 7 {
// PERFECT-NEXT:         %1 = affine.load %arg3[%arg10, %arg11] : memref<?x7xi32>
// PERFECT-NEXT:         affine.store %1, %arg7[%arg11] : memref<?xi32>
// PERFECT-NEXT:       }
// PERFECT-NEXT:     }
// PERFECT-NEXT:     affine.for %arg10 = 0 to 4 {
// PERFECT-NEXT:       affine.for %arg11 = 0 to 9 {
// PERFECT-NEXT:         %1 = affine.load %arg4[%arg10, %arg11] : memref<?x9xi32>
// PERFECT-NEXT:         %2 = affine.load %arg7[%arg11] : memref<?xi32>
// PERFECT-NEXT:         %3 = arith.addi %1, %2 : i32
// PERFECT-NEXT:         affine.store %3, %arg8[%arg11] : memref<?xi32>
// PERFECT-NEXT:       }
// PERFECT-NEXT:     }
// PERFECT-NEXT:     %0 = affine.load %arg9[0] : memref<?xi32>
// PERFECT-NEXT:     return %0 : i32
// PERFECT-NEXT:   }
// PERFECT-NEXT: }

// TASKFLOW:      module {
// TASKFLOW-NEXT:   func.func @_Z21pureNestedLoopExamplePA8_A6_iPA8_A5_iS4_PA7_iPA9_iPiS9_S9_S9_S9_(%arg0: memref<?x8x6xi32>, %arg1: memref<?x8x5xi32>, %arg2: memref<?x8x5xi32>, %arg3: memref<?x7xi32>, %arg4: memref<?x9xi32>, %arg5: memref<?xi32>, %arg6: memref<?xi32>, %arg7: memref<?xi32>, %arg8: memref<?xi32>, %arg9: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// TASKFLOW-NEXT:     %dependency_write_out = taskflow.task @Task_0 dependency_read_in(%arg0 : memref<?x8x6xi32>) dependency_write_in(%arg5 : memref<?xi32>) [original_read_memrefs(%arg0 : memref<?x8x6xi32>), original_write_memrefs(%arg5 : memref<?xi32>)] : (memref<?x8x6xi32>, memref<?xi32>) -> (memref<?xi32>) {
// TASKFLOW-NEXT:     ^bb0(%arg10: memref<?x8x6xi32>, %arg11: memref<?xi32>):
// TASKFLOW-NEXT:       affine.for %arg12 = 0 to 4 {
// TASKFLOW-NEXT:         affine.for %arg13 = 0 to 8 {
// TASKFLOW-NEXT:           affine.for %arg14 = 0 to 6 {
// TASKFLOW-NEXT:             %1 = affine.load %arg10[%arg12, %arg13, %arg14] : memref<?x8x6xi32>
// TASKFLOW-NEXT:             affine.store %1, %arg11[%arg14] : memref<?xi32>
// TASKFLOW-NEXT:           }
// TASKFLOW-NEXT:         }
// TASKFLOW-NEXT:       }
// TASKFLOW-NEXT:       taskflow.yield writes(%arg11 : memref<?xi32>)
// TASKFLOW-NEXT:     }
// TASKFLOW-NEXT:     %dependency_write_out_0 = taskflow.task @Task_1 dependency_read_in(%arg1, %arg2 : memref<?x8x5xi32>, memref<?x8x5xi32>) dependency_write_in(%arg6 : memref<?xi32>) [original_read_memrefs(%arg1, %arg2 : memref<?x8x5xi32>, memref<?x8x5xi32>), original_write_memrefs(%arg6 : memref<?xi32>)] : (memref<?x8x5xi32>, memref<?x8x5xi32>, memref<?xi32>) -> (memref<?xi32>) {
// TASKFLOW-NEXT:     ^bb0(%arg10: memref<?x8x5xi32>, %arg11: memref<?x8x5xi32>, %arg12: memref<?xi32>):
// TASKFLOW-NEXT:       affine.for %arg13 = 0 to 4 {
// TASKFLOW-NEXT:         affine.for %arg14 = 0 to 8 {
// TASKFLOW-NEXT:           affine.for %arg15 = 0 to 5 {
// TASKFLOW-NEXT:             %1 = affine.load %arg10[%arg13, %arg14, %arg15] : memref<?x8x5xi32>
// TASKFLOW-NEXT:             %2 = affine.load %arg11[%arg13, %arg14, %arg15] : memref<?x8x5xi32>
// TASKFLOW-NEXT:             %3 = arith.addi %1, %2 : i32
// TASKFLOW-NEXT:             affine.store %3, %arg12[%arg15] : memref<?xi32>
// TASKFLOW-NEXT:           }
// TASKFLOW-NEXT:         }
// TASKFLOW-NEXT:       }
// TASKFLOW-NEXT:       taskflow.yield writes(%arg12 : memref<?xi32>)
// TASKFLOW-NEXT:     }
// TASKFLOW-NEXT:     %dependency_write_out_1 = taskflow.task @Task_2 dependency_read_in(%dependency_write_out, %dependency_write_out_0, %arg9 : memref<?xi32>, memref<?xi32>, memref<?xi32>) dependency_write_in(%arg9 : memref<?xi32>) [original_read_memrefs(%arg5, %arg6, %arg9 : memref<?xi32>, memref<?xi32>, memref<?xi32>), original_write_memrefs(%arg9 : memref<?xi32>)] : (memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) -> (memref<?xi32>) {
// TASKFLOW-NEXT:     ^bb0(%arg10: memref<?xi32>, %arg11: memref<?xi32>, %arg12: memref<?xi32>, %arg13: memref<?xi32>):
// TASKFLOW-NEXT:       affine.for %arg14 = 0 to 4 {
// TASKFLOW-NEXT:         affine.for %arg15 = 0 to 8 {
// TASKFLOW-NEXT:           affine.for %arg16 = 0 to 6 {
// TASKFLOW-NEXT:             %1 = affine.load %arg10[%arg16] : memref<?xi32>
// TASKFLOW-NEXT:             %2 = affine.load %arg11[%arg16] : memref<?xi32>
// TASKFLOW-NEXT:             %3 = arith.addi %1, %2 : i32
// TASKFLOW-NEXT:             %4 = affine.load %arg13[0] : memref<?xi32>
// TASKFLOW-NEXT:             %5 = arith.addi %4, %3 : i32
// TASKFLOW-NEXT:             affine.store %5, %arg13[0] : memref<?xi32>
// TASKFLOW-NEXT:           }
// TASKFLOW-NEXT:         }
// TASKFLOW-NEXT:       }
// TASKFLOW-NEXT:       taskflow.yield writes(%arg13 : memref<?xi32>)
// TASKFLOW-NEXT:     }
// TASKFLOW-NEXT:     %dependency_write_out_2 = taskflow.task @Task_3 dependency_read_in(%arg3 : memref<?x7xi32>) dependency_write_in(%arg7 : memref<?xi32>) [original_read_memrefs(%arg3 : memref<?x7xi32>), original_write_memrefs(%arg7 : memref<?xi32>)] : (memref<?x7xi32>, memref<?xi32>) -> (memref<?xi32>) {
// TASKFLOW-NEXT:     ^bb0(%arg10: memref<?x7xi32>, %arg11: memref<?xi32>):
// TASKFLOW-NEXT:       affine.for %arg12 = 0 to 4 {
// TASKFLOW-NEXT:         affine.for %arg13 = 0 to 7 {
// TASKFLOW-NEXT:           %1 = affine.load %arg10[%arg12, %arg13] : memref<?x7xi32>
// TASKFLOW-NEXT:           affine.store %1, %arg11[%arg13] : memref<?xi32>
// TASKFLOW-NEXT:         }
// TASKFLOW-NEXT:       }
// TASKFLOW-NEXT:       taskflow.yield writes(%arg11 : memref<?xi32>)
// TASKFLOW-NEXT:     }
// TASKFLOW-NEXT:     %dependency_write_out_3 = taskflow.task @Task_4 dependency_read_in(%arg4, %dependency_write_out_2 : memref<?x9xi32>, memref<?xi32>) dependency_write_in(%arg8 : memref<?xi32>) [original_read_memrefs(%arg4, %arg7 : memref<?x9xi32>, memref<?xi32>), original_write_memrefs(%arg8 : memref<?xi32>)] : (memref<?x9xi32>, memref<?xi32>, memref<?xi32>) -> (memref<?xi32>) {
// TASKFLOW-NEXT:     ^bb0(%arg10: memref<?x9xi32>, %arg11: memref<?xi32>, %arg12: memref<?xi32>):
// TASKFLOW-NEXT:       affine.for %arg13 = 0 to 4 {
// TASKFLOW-NEXT:         affine.for %arg14 = 0 to 9 {
// TASKFLOW-NEXT:           %1 = affine.load %arg10[%arg13, %arg14] : memref<?x9xi32>
// TASKFLOW-NEXT:           %2 = affine.load %arg11[%arg14] : memref<?xi32>
// TASKFLOW-NEXT:           %3 = arith.addi %1, %2 : i32
// TASKFLOW-NEXT:           affine.store %3, %arg12[%arg14] : memref<?xi32>
// TASKFLOW-NEXT:         }
// TASKFLOW-NEXT:       }
// TASKFLOW-NEXT:       taskflow.yield writes(%arg12 : memref<?xi32>)
// TASKFLOW-NEXT:     }
// TASKFLOW-NEXT:     %0 = affine.load %dependency_write_out_1[0] : memref<?xi32>
// TASKFLOW-NEXT:     return %0 : i32
// TASKFLOW-NEXT:   }
// TASKFLOW-NEXT: }

// STREAM:      module {
// STREAM-NEXT:   func.func @_Z21pureNestedLoopExamplePA8_A6_iPA8_A5_iS4_PA7_iPA9_iPiS9_S9_S9_S9_(%arg0: memref<?x8x6xi32>, %arg1: memref<?x8x5xi32>, %arg2: memref<?x8x5xi32>, %arg3: memref<?x7xi32>, %arg4: memref<?x9xi32>, %arg5: memref<?xi32>, %arg6: memref<?xi32>, %arg7: memref<?xi32>, %arg8: memref<?xi32>, %arg9: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// STREAM-NEXT:     %dependency_write_out = taskflow.task @Task_1 dependency_read_in(%arg1, %arg2 : memref<?x8x5xi32>, memref<?x8x5xi32>) dependency_write_in(%arg6 : memref<?xi32>) [original_read_memrefs(%arg1, %arg2 : memref<?x8x5xi32>, memref<?x8x5xi32>), original_write_memrefs(%arg6 : memref<?xi32>)] : (memref<?x8x5xi32>, memref<?x8x5xi32>, memref<?xi32>) -> (memref<?xi32>) {
// STREAM-NEXT:     ^bb0(%arg10: memref<?x8x5xi32>, %arg11: memref<?x8x5xi32>, %arg12: memref<?xi32>):
// STREAM-NEXT:       affine.for %arg13 = 0 to 4 {
// STREAM-NEXT:         affine.for %arg14 = 0 to 8 {
// STREAM-NEXT:           affine.for %arg15 = 0 to 5 {
// STREAM-NEXT:             %1 = affine.load %arg10[%arg13, %arg14, %arg15] : memref<?x8x5xi32>
// STREAM-NEXT:             %2 = affine.load %arg11[%arg13, %arg14, %arg15] : memref<?x8x5xi32>
// STREAM-NEXT:             %3 = arith.addi %1, %2 : i32
// STREAM-NEXT:             affine.store %3, %arg12[%arg15] : memref<?xi32>
// STREAM-NEXT:           }
// STREAM-NEXT:         }
// STREAM-NEXT:       }
// STREAM-NEXT:       taskflow.yield writes(%arg12 : memref<?xi32>)
// STREAM-NEXT:     }
// STREAM-NEXT:     %dependency_read_out:3, %dependency_write_out_0 = taskflow.task @Task_0_Task_2_fused dependency_read_in(%arg0, %dependency_write_out, %arg9 : memref<?x8x6xi32>, memref<?xi32>, memref<?xi32>) dependency_write_in(%arg9 : memref<?xi32>) [original_read_memrefs(%arg0, %arg6, %arg9 : memref<?x8x6xi32>, memref<?xi32>, memref<?xi32>), original_write_memrefs(%arg9 : memref<?xi32>)] : (memref<?x8x6xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) -> (memref<?x8x6xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) {
// STREAM-NEXT:     ^bb0(%arg10: memref<?x8x6xi32>, %arg11: memref<?xi32>, %arg12: memref<?xi32>, %arg13: memref<?xi32>):
// STREAM-NEXT:       affine.for %arg14 = 0 to 4 {
// STREAM-NEXT:         affine.for %arg15 = 0 to 8 {
// STREAM-NEXT:           affine.for %arg16 = 0 to 6 {
// STREAM-NEXT:             %1 = affine.load %arg10[%arg14, %arg15, %arg16] : memref<?x8x6xi32>
// STREAM-NEXT:             %2 = affine.load %arg11[%arg16] : memref<?xi32>
// STREAM-NEXT:             %3 = arith.addi %1, %2 : i32
// STREAM-NEXT:             %4 = affine.load %arg12[0] : memref<?xi32>
// STREAM-NEXT:             %5 = arith.addi %4, %3 : i32
// STREAM-NEXT:             affine.store %5, %arg12[0] : memref<?xi32>
// STREAM-NEXT:           }
// STREAM-NEXT:         }
// STREAM-NEXT:       }
// STREAM-NEXT:       taskflow.yield reads(%arg10, %arg11, %arg12 : memref<?x8x6xi32>, memref<?xi32>, memref<?xi32>) writes(%arg12 : memref<?xi32>)
// STREAM-NEXT:     }
// STREAM-NEXT:     %dependency_write_out_1 = taskflow.task @Task_3 dependency_read_in(%arg3 : memref<?x7xi32>) dependency_write_in(%arg7 : memref<?xi32>) [original_read_memrefs(%arg3 : memref<?x7xi32>), original_write_memrefs(%arg7 : memref<?xi32>)] : (memref<?x7xi32>, memref<?xi32>) -> (memref<?xi32>) {
// STREAM-NEXT:     ^bb0(%arg10: memref<?x7xi32>, %arg11: memref<?xi32>):
// STREAM-NEXT:       affine.for %arg12 = 0 to 4 {
// STREAM-NEXT:         affine.for %arg13 = 0 to 7 {
// STREAM-NEXT:           %1 = affine.load %arg10[%arg12, %arg13] : memref<?x7xi32>
// STREAM-NEXT:           affine.store %1, %arg11[%arg13] : memref<?xi32>
// STREAM-NEXT:         }
// STREAM-NEXT:       }
// STREAM-NEXT:       taskflow.yield writes(%arg11 : memref<?xi32>)
// STREAM-NEXT:     }
// STREAM-NEXT:     %dependency_write_out_2 = taskflow.task @Task_4 dependency_read_in(%arg4, %dependency_write_out_1 : memref<?x9xi32>, memref<?xi32>) dependency_write_in(%arg8 : memref<?xi32>) [original_read_memrefs(%arg4, %arg7 : memref<?x9xi32>, memref<?xi32>), original_write_memrefs(%arg8 : memref<?xi32>)] : (memref<?x9xi32>, memref<?xi32>, memref<?xi32>) -> (memref<?xi32>) {
// STREAM-NEXT:     ^bb0(%arg10: memref<?x9xi32>, %arg11: memref<?xi32>, %arg12: memref<?xi32>):
// STREAM-NEXT:       affine.for %arg13 = 0 to 4 {
// STREAM-NEXT:         affine.for %arg14 = 0 to 9 {
// STREAM-NEXT:           %1 = affine.load %arg10[%arg13, %arg14] : memref<?x9xi32>
// STREAM-NEXT:           %2 = affine.load %arg11[%arg14] : memref<?xi32>
// STREAM-NEXT:           %3 = arith.addi %1, %2 : i32
// STREAM-NEXT:           affine.store %3, %arg12[%arg14] : memref<?xi32>
// STREAM-NEXT:         }
// STREAM-NEXT:       }
// STREAM-NEXT:       taskflow.yield writes(%arg12 : memref<?xi32>)
// STREAM-NEXT:     }
// STREAM-NEXT:     %0 = affine.load %dependency_write_out_0[0] : memref<?xi32>
// STREAM-NEXT:     return %0 : i32
// STREAM-NEXT:   }
// STREAM-NEXT: }

// KERNEL:      module {
// KERNEL-NEXT:   func.func @_Z21pureNestedLoopExamplePA8_A6_iPA8_A5_iS4_PA7_iPA9_iPiS9_S9_S9_S9_(%arg0: memref<?x8x6xi32>, %arg1: memref<?x8x5xi32>, %arg2: memref<?x8x5xi32>, %arg3: memref<?x7xi32>, %arg4: memref<?x9xi32>, %arg5: memref<?xi32>, %arg6: memref<?xi32>, %arg7: memref<?xi32>, %arg8: memref<?xi32>, %arg9: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// KERNEL-NEXT:     %dependency_write_out = taskflow.task @Task_0 dependency_read_in(%arg0 : memref<?x8x6xi32>) dependency_write_in(%arg5 : memref<?xi32>) [original_read_memrefs(%arg0 : memref<?x8x6xi32>), original_write_memrefs(%arg5 : memref<?xi32>)] : (memref<?x8x6xi32>, memref<?xi32>) -> (memref<?xi32>) {
// KERNEL-NEXT:     ^bb0(%arg10: memref<?x8x6xi32>, %arg11: memref<?xi32>):
// KERNEL-NEXT:       affine.for %arg12 = 0 to 4 {
// KERNEL-NEXT:         affine.for %arg13 = 0 to 8 {
// KERNEL-NEXT:           neura.kernel inputs(%arg10, %arg12, %arg13, %arg11 : memref<?x8x6xi32>, index, index, memref<?xi32>) {
// KERNEL-NEXT:           ^bb0(%arg14: memref<?x8x6xi32>, %arg15: index, %arg16: index, %arg17: memref<?xi32>):
// KERNEL-NEXT:             %c0 = arith.constant 0 : index
// KERNEL-NEXT:             %c6 = arith.constant 6 : index
// KERNEL-NEXT:             %c1 = arith.constant 1 : index
// KERNEL-NEXT:             scf.for %arg18 = %c0 to %c6 step %c1 {
// KERNEL-NEXT:               %1 = memref.load %arg14[%arg15, %arg16, %arg18] : memref<?x8x6xi32>
// KERNEL-NEXT:               memref.store %1, %arg17[%arg18] : memref<?xi32>
// KERNEL-NEXT:             }
// KERNEL-NEXT:             neura.yield
// KERNEL-NEXT:           }
// KERNEL-NEXT:         }
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield writes(%arg11 : memref<?xi32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %dependency_write_out_0 = taskflow.task @Task_1 dependency_read_in(%arg1, %arg2 : memref<?x8x5xi32>, memref<?x8x5xi32>) dependency_write_in(%arg6 : memref<?xi32>) [original_read_memrefs(%arg1, %arg2 : memref<?x8x5xi32>, memref<?x8x5xi32>), original_write_memrefs(%arg6 : memref<?xi32>)] : (memref<?x8x5xi32>, memref<?x8x5xi32>, memref<?xi32>) -> (memref<?xi32>) {
// KERNEL-NEXT:     ^bb0(%arg10: memref<?x8x5xi32>, %arg11: memref<?x8x5xi32>, %arg12: memref<?xi32>):
// KERNEL-NEXT:       affine.for %arg13 = 0 to 4 {
// KERNEL-NEXT:         affine.for %arg14 = 0 to 8 {
// KERNEL-NEXT:           neura.kernel inputs(%arg10, %arg13, %arg14, %arg11, %arg12 : memref<?x8x5xi32>, index, index, memref<?x8x5xi32>, memref<?xi32>) {
// KERNEL-NEXT:           ^bb0(%arg15: memref<?x8x5xi32>, %arg16: index, %arg17: index, %arg18: memref<?x8x5xi32>, %arg19: memref<?xi32>):
// KERNEL-NEXT:             %c0 = arith.constant 0 : index
// KERNEL-NEXT:             %c5 = arith.constant 5 : index
// KERNEL-NEXT:             %c1 = arith.constant 1 : index
// KERNEL-NEXT:             scf.for %arg20 = %c0 to %c5 step %c1 {
// KERNEL-NEXT:               %1 = memref.load %arg15[%arg16, %arg17, %arg20] : memref<?x8x5xi32>
// KERNEL-NEXT:               %2 = memref.load %arg18[%arg16, %arg17, %arg20] : memref<?x8x5xi32>
// KERNEL-NEXT:               %3 = arith.addi %1, %2 : i32
// KERNEL-NEXT:               memref.store %3, %arg19[%arg20] : memref<?xi32>
// KERNEL-NEXT:             }
// KERNEL-NEXT:             neura.yield
// KERNEL-NEXT:           }
// KERNEL-NEXT:         }
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield writes(%arg12 : memref<?xi32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %dependency_write_out_1 = taskflow.task @Task_2 dependency_read_in(%dependency_write_out, %dependency_write_out_0, %arg9 : memref<?xi32>, memref<?xi32>, memref<?xi32>) dependency_write_in(%arg9 : memref<?xi32>) [original_read_memrefs(%arg5, %arg6, %arg9 : memref<?xi32>, memref<?xi32>, memref<?xi32>), original_write_memrefs(%arg9 : memref<?xi32>)] : (memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) -> (memref<?xi32>) {
// KERNEL-NEXT:     ^bb0(%arg10: memref<?xi32>, %arg11: memref<?xi32>, %arg12: memref<?xi32>, %arg13: memref<?xi32>):
// KERNEL-NEXT:       affine.for %arg14 = 0 to 4 {
// KERNEL-NEXT:         affine.for %arg15 = 0 to 8 {
// KERNEL-NEXT:           neura.kernel inputs(%arg10, %arg11, %arg13 : memref<?xi32>, memref<?xi32>, memref<?xi32>) {
// KERNEL-NEXT:           ^bb0(%arg16: memref<?xi32>, %arg17: memref<?xi32>, %arg18: memref<?xi32>):
// KERNEL-NEXT:             %c0 = arith.constant 0 : index
// KERNEL-NEXT:             %c6 = arith.constant 6 : index
// KERNEL-NEXT:             %c1 = arith.constant 1 : index
// KERNEL-NEXT:             scf.for %arg19 = %c0 to %c6 step %c1 {
// KERNEL-NEXT:               %1 = memref.load %arg16[%arg19] : memref<?xi32>
// KERNEL-NEXT:               %2 = memref.load %arg17[%arg19] : memref<?xi32>
// KERNEL-NEXT:               %3 = arith.addi %1, %2 : i32
// KERNEL-NEXT:               %4 = memref.load %arg18[%c0] : memref<?xi32>
// KERNEL-NEXT:               %5 = arith.addi %4, %3 : i32
// KERNEL-NEXT:               memref.store %5, %arg18[%c0] : memref<?xi32>
// KERNEL-NEXT:             }
// KERNEL-NEXT:             neura.yield
// KERNEL-NEXT:           }
// KERNEL-NEXT:         }
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield writes(%arg13 : memref<?xi32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %dependency_write_out_2 = taskflow.task @Task_3 dependency_read_in(%arg3 : memref<?x7xi32>) dependency_write_in(%arg7 : memref<?xi32>) [original_read_memrefs(%arg3 : memref<?x7xi32>), original_write_memrefs(%arg7 : memref<?xi32>)] : (memref<?x7xi32>, memref<?xi32>) -> (memref<?xi32>) {
// KERNEL-NEXT:     ^bb0(%arg10: memref<?x7xi32>, %arg11: memref<?xi32>):
// KERNEL-NEXT:       affine.for %arg12 = 0 to 4 {
// KERNEL-NEXT:         neura.kernel inputs(%arg10, %arg12, %arg11 : memref<?x7xi32>, index, memref<?xi32>) {
// KERNEL-NEXT:         ^bb0(%arg13: memref<?x7xi32>, %arg14: index, %arg15: memref<?xi32>):
// KERNEL-NEXT:           %c0 = arith.constant 0 : index
// KERNEL-NEXT:           %c7 = arith.constant 7 : index
// KERNEL-NEXT:           %c1 = arith.constant 1 : index
// KERNEL-NEXT:           scf.for %arg16 = %c0 to %c7 step %c1 {
// KERNEL-NEXT:             %1 = memref.load %arg13[%arg14, %arg16] : memref<?x7xi32>
// KERNEL-NEXT:             memref.store %1, %arg15[%arg16] : memref<?xi32>
// KERNEL-NEXT:           }
// KERNEL-NEXT:           neura.yield
// KERNEL-NEXT:         }
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield writes(%arg11 : memref<?xi32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %dependency_write_out_3 = taskflow.task @Task_4 dependency_read_in(%arg4, %dependency_write_out_2 : memref<?x9xi32>, memref<?xi32>) dependency_write_in(%arg8 : memref<?xi32>) [original_read_memrefs(%arg4, %arg7 : memref<?x9xi32>, memref<?xi32>), original_write_memrefs(%arg8 : memref<?xi32>)] : (memref<?x9xi32>, memref<?xi32>, memref<?xi32>) -> (memref<?xi32>) {
// KERNEL-NEXT:     ^bb0(%arg10: memref<?x9xi32>, %arg11: memref<?xi32>, %arg12: memref<?xi32>):
// KERNEL-NEXT:       affine.for %arg13 = 0 to 4 {
// KERNEL-NEXT:         neura.kernel inputs(%arg10, %arg13, %arg11, %arg12 : memref<?x9xi32>, index, memref<?xi32>, memref<?xi32>) {
// KERNEL-NEXT:         ^bb0(%arg14: memref<?x9xi32>, %arg15: index, %arg16: memref<?xi32>, %arg17: memref<?xi32>):
// KERNEL-NEXT:           %c0 = arith.constant 0 : index
// KERNEL-NEXT:           %c9 = arith.constant 9 : index
// KERNEL-NEXT:           %c1 = arith.constant 1 : index
// KERNEL-NEXT:           scf.for %arg18 = %c0 to %c9 step %c1 {
// KERNEL-NEXT:             %1 = memref.load %arg14[%arg15, %arg18] : memref<?x9xi32>
// KERNEL-NEXT:             %2 = memref.load %arg16[%arg18] : memref<?xi32>
// KERNEL-NEXT:             %3 = arith.addi %1, %2 : i32
// KERNEL-NEXT:             memref.store %3, %arg17[%arg18] : memref<?xi32>
// KERNEL-NEXT:           }
// KERNEL-NEXT:           neura.yield
// KERNEL-NEXT:         }
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield writes(%arg12 : memref<?xi32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %0 = affine.load %dependency_write_out_1[0] : memref<?xi32>
// KERNEL-NEXT:     return %0 : i32
// KERNEL-NEXT:   }
// KERNEL-NEXT: }

// HYPERBLOCK:      module {
// HYPERBLOCK-NEXT:   func.func @_Z21pureNestedLoopExamplePA8_A6_iPA8_A5_iS4_PA7_iPA9_iPiS9_S9_S9_S9_(%arg0: memref<?x8x6xi32>, %arg1: memref<?x8x5xi32>, %arg2: memref<?x8x5xi32>, %arg3: memref<?x7xi32>, %arg4: memref<?x9xi32>, %arg5: memref<?xi32>, %arg6: memref<?xi32>, %arg7: memref<?xi32>, %arg8: memref<?xi32>, %arg9: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// HYPERBLOCK-NEXT:     %dependency_write_out = taskflow.task @Task_0 dependency_read_in(%arg0 : memref<?x8x6xi32>) dependency_write_in(%arg5 : memref<?xi32>) [original_read_memrefs(%arg0 : memref<?x8x6xi32>), original_write_memrefs(%arg5 : memref<?xi32>)] : (memref<?x8x6xi32>, memref<?xi32>) -> (memref<?xi32>) {
// HYPERBLOCK-NEXT:     ^bb0(%arg10: memref<?x8x6xi32>, %arg11: memref<?xi32>):
// HYPERBLOCK-NEXT:       %c0 = arith.constant 0 : index
// HYPERBLOCK-NEXT:       %c4 = arith.constant 4 : index
// HYPERBLOCK-NEXT:       %c1 = arith.constant 1 : index
// HYPERBLOCK-NEXT:       %1 = taskflow.counter from %c0 to %c4 step %c1 : index
// HYPERBLOCK-NEXT:       %c8 = arith.constant 8 : index
// HYPERBLOCK-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0 to %c8 step %c1 : index
// HYPERBLOCK-NEXT:       %c6 = arith.constant 6 : index
// HYPERBLOCK-NEXT:       %3 = taskflow.counter parent(%2 : index) from %c0 to %c6 step %c1 : index
// HYPERBLOCK-NEXT:       "taskflow.hyperblock"(%1, %2, %3) <{operandSegmentSizes = array<i32: 3, 0>}> ({
// HYPERBLOCK-NEXT:       ^bb0(%arg12: index, %arg13: index, %arg14: index):
// HYPERBLOCK-NEXT:         %4 = memref.load %arg10[%arg12, %arg13, %arg14] : memref<?x8x6xi32>
// HYPERBLOCK-NEXT:         memref.store %4, %arg11[%arg14] : memref<?xi32>
// HYPERBLOCK-NEXT:         taskflow.hyperblock.yield
// HYPERBLOCK-NEXT:       }) : (index, index, index) -> ()
// HYPERBLOCK-NEXT:       taskflow.yield writes(%arg11 : memref<?xi32>)
// HYPERBLOCK-NEXT:     }
// HYPERBLOCK-NEXT:     %dependency_write_out_0 = taskflow.task @Task_1 dependency_read_in(%arg1, %arg2 : memref<?x8x5xi32>, memref<?x8x5xi32>) dependency_write_in(%arg6 : memref<?xi32>) [original_read_memrefs(%arg1, %arg2 : memref<?x8x5xi32>, memref<?x8x5xi32>), original_write_memrefs(%arg6 : memref<?xi32>)] : (memref<?x8x5xi32>, memref<?x8x5xi32>, memref<?xi32>) -> (memref<?xi32>) {
// HYPERBLOCK-NEXT:     ^bb0(%arg10: memref<?x8x5xi32>, %arg11: memref<?x8x5xi32>, %arg12: memref<?xi32>):
// HYPERBLOCK-NEXT:       %c0 = arith.constant 0 : index
// HYPERBLOCK-NEXT:       %c4 = arith.constant 4 : index
// HYPERBLOCK-NEXT:       %c1 = arith.constant 1 : index
// HYPERBLOCK-NEXT:       %1 = taskflow.counter from %c0 to %c4 step %c1 : index
// HYPERBLOCK-NEXT:       %c8 = arith.constant 8 : index
// HYPERBLOCK-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0 to %c8 step %c1 : index
// HYPERBLOCK-NEXT:       %c5 = arith.constant 5 : index
// HYPERBLOCK-NEXT:       %3 = taskflow.counter parent(%2 : index) from %c0 to %c5 step %c1 : index
// HYPERBLOCK-NEXT:       "taskflow.hyperblock"(%1, %2, %3) <{operandSegmentSizes = array<i32: 3, 0>}> ({
// HYPERBLOCK-NEXT:       ^bb0(%arg13: index, %arg14: index, %arg15: index):
// HYPERBLOCK-NEXT:         %4 = memref.load %arg10[%arg13, %arg14, %arg15] : memref<?x8x5xi32>
// HYPERBLOCK-NEXT:         %5 = memref.load %arg11[%arg13, %arg14, %arg15] : memref<?x8x5xi32>
// HYPERBLOCK-NEXT:         %6 = arith.addi %4, %5 : i32
// HYPERBLOCK-NEXT:         memref.store %6, %arg12[%arg15] : memref<?xi32>
// HYPERBLOCK-NEXT:         taskflow.hyperblock.yield
// HYPERBLOCK-NEXT:       }) : (index, index, index) -> ()
// HYPERBLOCK-NEXT:       taskflow.yield writes(%arg12 : memref<?xi32>)
// HYPERBLOCK-NEXT:     }
// HYPERBLOCK-NEXT:     %dependency_write_out_1 = taskflow.task @Task_2 dependency_read_in(%dependency_write_out, %dependency_write_out_0, %arg9 : memref<?xi32>, memref<?xi32>, memref<?xi32>) dependency_write_in(%arg9 : memref<?xi32>) [original_read_memrefs(%arg5, %arg6, %arg9 : memref<?xi32>, memref<?xi32>, memref<?xi32>), original_write_memrefs(%arg9 : memref<?xi32>)] : (memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) -> (memref<?xi32>) {
// HYPERBLOCK-NEXT:     ^bb0(%arg10: memref<?xi32>, %arg11: memref<?xi32>, %arg12: memref<?xi32>, %arg13: memref<?xi32>):
// HYPERBLOCK-NEXT:       %c0 = arith.constant 0 : index
// HYPERBLOCK-NEXT:       %c4 = arith.constant 4 : index
// HYPERBLOCK-NEXT:       %c1 = arith.constant 1 : index
// HYPERBLOCK-NEXT:       %1 = taskflow.counter from %c0 to %c4 step %c1 : index
// HYPERBLOCK-NEXT:       %c8 = arith.constant 8 : index
// HYPERBLOCK-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0 to %c8 step %c1 : index
// HYPERBLOCK-NEXT:       %c6 = arith.constant 6 : index
// HYPERBLOCK-NEXT:       %3 = taskflow.counter parent(%2 : index) from %c0 to %c6 step %c1 : index
// HYPERBLOCK-NEXT:       "taskflow.hyperblock"(%3) <{operandSegmentSizes = array<i32: 1, 0>}> ({
// HYPERBLOCK-NEXT:       ^bb0(%arg14: index):
// HYPERBLOCK-NEXT:         %4 = memref.load %arg10[%arg14] : memref<?xi32>
// HYPERBLOCK-NEXT:         %5 = memref.load %arg11[%arg14] : memref<?xi32>
// HYPERBLOCK-NEXT:         %6 = arith.addi %4, %5 : i32
// HYPERBLOCK-NEXT:         %c0_4 = arith.constant 0 : index
// HYPERBLOCK-NEXT:         %7 = memref.load %arg13[%c0_4] : memref<?xi32>
// HYPERBLOCK-NEXT:         %8 = arith.addi %7, %6 : i32
// HYPERBLOCK-NEXT:         %c0_5 = arith.constant 0 : index
// HYPERBLOCK-NEXT:         memref.store %8, %arg13[%c0_5] : memref<?xi32>
// HYPERBLOCK-NEXT:         taskflow.hyperblock.yield
// HYPERBLOCK-NEXT:       }) : (index) -> ()
// HYPERBLOCK-NEXT:       taskflow.yield writes(%arg13 : memref<?xi32>)
// HYPERBLOCK-NEXT:     }
// HYPERBLOCK-NEXT:     %dependency_write_out_2 = taskflow.task @Task_3 dependency_read_in(%arg3 : memref<?x7xi32>) dependency_write_in(%arg7 : memref<?xi32>) [original_read_memrefs(%arg3 : memref<?x7xi32>), original_write_memrefs(%arg7 : memref<?xi32>)] : (memref<?x7xi32>, memref<?xi32>) -> (memref<?xi32>) {
// HYPERBLOCK-NEXT:     ^bb0(%arg10: memref<?x7xi32>, %arg11: memref<?xi32>):
// HYPERBLOCK-NEXT:       %c0 = arith.constant 0 : index
// HYPERBLOCK-NEXT:       %c4 = arith.constant 4 : index
// HYPERBLOCK-NEXT:       %c1 = arith.constant 1 : index
// HYPERBLOCK-NEXT:       %1 = taskflow.counter from %c0 to %c4 step %c1 : index
// HYPERBLOCK-NEXT:       %c7 = arith.constant 7 : index
// HYPERBLOCK-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0 to %c7 step %c1 : index
// HYPERBLOCK-NEXT:       "taskflow.hyperblock"(%1, %2) <{operandSegmentSizes = array<i32: 2, 0>}> ({
// HYPERBLOCK-NEXT:       ^bb0(%arg12: index, %arg13: index):
// HYPERBLOCK-NEXT:         %3 = memref.load %arg10[%arg12, %arg13] : memref<?x7xi32>
// HYPERBLOCK-NEXT:         memref.store %3, %arg11[%arg13] : memref<?xi32>
// HYPERBLOCK-NEXT:         taskflow.hyperblock.yield
// HYPERBLOCK-NEXT:       }) : (index, index) -> ()
// HYPERBLOCK-NEXT:       taskflow.yield writes(%arg11 : memref<?xi32>)
// HYPERBLOCK-NEXT:     }
// HYPERBLOCK-NEXT:     %dependency_write_out_3 = taskflow.task @Task_4 dependency_read_in(%arg4, %dependency_write_out_2 : memref<?x9xi32>, memref<?xi32>) dependency_write_in(%arg8 : memref<?xi32>) [original_read_memrefs(%arg4, %arg7 : memref<?x9xi32>, memref<?xi32>), original_write_memrefs(%arg8 : memref<?xi32>)] : (memref<?x9xi32>, memref<?xi32>, memref<?xi32>) -> (memref<?xi32>) {
// HYPERBLOCK-NEXT:     ^bb0(%arg10: memref<?x9xi32>, %arg11: memref<?xi32>, %arg12: memref<?xi32>):
// HYPERBLOCK-NEXT:       %c0 = arith.constant 0 : index
// HYPERBLOCK-NEXT:       %c4 = arith.constant 4 : index
// HYPERBLOCK-NEXT:       %c1 = arith.constant 1 : index
// HYPERBLOCK-NEXT:       %1 = taskflow.counter from %c0 to %c4 step %c1 : index
// HYPERBLOCK-NEXT:       %c9 = arith.constant 9 : index
// HYPERBLOCK-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0 to %c9 step %c1 : index
// HYPERBLOCK-NEXT:       "taskflow.hyperblock"(%1, %2) <{operandSegmentSizes = array<i32: 2, 0>}> ({
// HYPERBLOCK-NEXT:       ^bb0(%arg13: index, %arg14: index):
// HYPERBLOCK-NEXT:         %3 = memref.load %arg10[%arg13, %arg14] : memref<?x9xi32>
// HYPERBLOCK-NEXT:         %4 = memref.load %arg11[%arg14] : memref<?xi32>
// HYPERBLOCK-NEXT:         %5 = arith.addi %3, %4 : i32
// HYPERBLOCK-NEXT:         memref.store %5, %arg12[%arg14] : memref<?xi32>
// HYPERBLOCK-NEXT:         taskflow.hyperblock.yield
// HYPERBLOCK-NEXT:       }) : (index, index) -> ()
// HYPERBLOCK-NEXT:       taskflow.yield writes(%arg12 : memref<?xi32>)
// HYPERBLOCK-NEXT:     }
// HYPERBLOCK-NEXT:     %0 = affine.load %dependency_write_out_1[0] : memref<?xi32>
// HYPERBLOCK-NEXT:     return %0 : i32
// HYPERBLOCK-NEXT:   }
// HYPERBLOCK-NEXT: }


// MAP-SPATIAL-TEMPORAL-4x4:     module {
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:  func.func @_Z21pureNestedLoopExamplePA8_A6_iPA8_A5_iS4_PA7_iPA9_iPiS9_S9_S9_S9_(%arg0: memref<?x8x6xi32>, %arg1: memref<?x8x5xi32>, %arg2: memref<?x8x5xi32>, %arg3: memref<?x7xi32>, %arg4: memref<?x9xi32>, %arg5: memref<?xi32>, %arg6: memref<?xi32>, %arg7: memref<?xi32>, %arg8: memref<?xi32>, %arg9: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:    %dependency_write_out = taskflow.task @Task_0 dependency_read_in(%arg0 : memref<?x8x6xi32>) dependency_write_in(%arg5 : memref<?xi32>) [original_read_memrefs(%arg0 : memref<?x8x6xi32>), original_write_memrefs(%arg5 : memref<?xi32>)] {profile_info = {duration = 1 : i32}, task_orchestration_info = {cgra_positions = [{col = 0 : i32, context_id = 0 : i32, row = 0 : i32}], read_sram_locations = [{col = 0 : i32, row = 0 : i32}], write_sram_locations = [{col = 0 : i32, row = 1 : i32}]}} : (memref<?x8x6xi32>, memref<?xi32>) -> (memref<?xi32>) {
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:    ^bb0(%arg10: memref<?x8x6xi32>, %arg11: memref<?xi32>):
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %c0 = arith.constant 0 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %c4 = arith.constant 4 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %c1 = arith.constant 1 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %1 = taskflow.counter from %c0 to %c4 step %c1 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %c8 = arith.constant 8 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %2 = taskflow.counter parent(%1 : index) from %c0 to %c8 step %c1 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %c6 = arith.constant 6 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %3 = taskflow.counter parent(%2 : index) from %c0 to %c6 step %c1 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      "taskflow.hyperblock"(%1, %2, %3) <{operandSegmentSizes = array<i32: 3, 0>}> ({
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      ^bb0(%arg12: index, %arg13: index, %arg14: index):
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:        %4 = memref.load %arg10[%arg12, %arg13, %arg14] : memref<?x8x6xi32>
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:        memref.store %4, %arg11[%arg14] : memref<?xi32>
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:        taskflow.hyperblock.yield
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      }) : (index, index, index) -> ()
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      taskflow.yield writes(%arg11 : memref<?xi32>)
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:    }
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:    %dependency_write_out_0 = taskflow.task @Task_1 dependency_read_in(%arg1, %arg2 : memref<?x8x5xi32>, memref<?x8x5xi32>) dependency_write_in(%arg6 : memref<?xi32>) [original_read_memrefs(%arg1, %arg2 : memref<?x8x5xi32>, memref<?x8x5xi32>), original_write_memrefs(%arg6 : memref<?xi32>)] {profile_info = {duration = 1 : i32}, task_orchestration_info = {cgra_positions = [{col = 1 : i32, context_id = 0 : i32, row = 0 : i32}], read_sram_locations = [{col = 1 : i32, row = 0 : i32}, {col = 1 : i32, row = 0 : i32}], write_sram_locations = [{col = 1 : i32, row = 1 : i32}]}} : (memref<?x8x5xi32>, memref<?x8x5xi32>, memref<?xi32>) -> (memref<?xi32>) {
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:    ^bb0(%arg10: memref<?x8x5xi32>, %arg11: memref<?x8x5xi32>, %arg12: memref<?xi32>):
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %c0 = arith.constant 0 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %c4 = arith.constant 4 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %c1 = arith.constant 1 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %1 = taskflow.counter from %c0 to %c4 step %c1 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %c8 = arith.constant 8 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %2 = taskflow.counter parent(%1 : index) from %c0 to %c8 step %c1 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %c5 = arith.constant 5 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %3 = taskflow.counter parent(%2 : index) from %c0 to %c5 step %c1 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      "taskflow.hyperblock"(%1, %2, %3) <{operandSegmentSizes = array<i32: 3, 0>}> ({
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      ^bb0(%arg13: index, %arg14: index, %arg15: index):
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:        %4 = memref.load %arg10[%arg13, %arg14, %arg15] : memref<?x8x5xi32>
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:        %5 = memref.load %arg11[%arg13, %arg14, %arg15] : memref<?x8x5xi32>
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:        %6 = arith.addi %4, %5 : i32
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:        memref.store %6, %arg12[%arg15] : memref<?xi32>
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:        taskflow.hyperblock.yield
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      }) : (index, index, index) -> ()
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      taskflow.yield writes(%arg12 : memref<?xi32>)
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:    }
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:    %dependency_write_out_1 = taskflow.task @Task_2 dependency_read_in(%dependency_write_out, %dependency_write_out_0, %arg9 : memref<?xi32>, memref<?xi32>, memref<?xi32>) dependency_write_in(%arg9 : memref<?xi32>) [original_read_memrefs(%arg5, %arg6, %arg9 : memref<?xi32>, memref<?xi32>, memref<?xi32>), original_write_memrefs(%arg9 : memref<?xi32>)] {profile_info = {duration = 1 : i32}, task_orchestration_info = {cgra_positions = [{col = 0 : i32, context_id = 0 : i32, row = 1 : i32}], read_sram_locations = [{col = 0 : i32, row = 1 : i32}, {col = 1 : i32, row = 1 : i32}, {col = 0 : i32, row = 1 : i32}], write_sram_locations = [{col = 0 : i32, row = 1 : i32}]}} : (memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) -> (memref<?xi32>) {
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:    ^bb0(%arg10: memref<?xi32>, %arg11: memref<?xi32>, %arg12: memref<?xi32>, %arg13: memref<?xi32>):
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %c0 = arith.constant 0 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %c4 = arith.constant 4 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %c1 = arith.constant 1 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %1 = taskflow.counter from %c0 to %c4 step %c1 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %c8 = arith.constant 8 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %2 = taskflow.counter parent(%1 : index) from %c0 to %c8 step %c1 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %c6 = arith.constant 6 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %3 = taskflow.counter parent(%2 : index) from %c0 to %c6 step %c1 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      "taskflow.hyperblock"(%3) <{operandSegmentSizes = array<i32: 1, 0>}> ({
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      ^bb0(%arg14: index):
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:        %4 = memref.load %arg10[%arg14] : memref<?xi32>
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:        %5 = memref.load %arg11[%arg14] : memref<?xi32>
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:        %6 = arith.addi %4, %5 : i32
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:        %c0_4 = arith.constant 0 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:        %7 = memref.load %arg13[%c0_4] : memref<?xi32>
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:        %8 = arith.addi %7, %6 : i32
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:        %c0_5 = arith.constant 0 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:        memref.store %8, %arg13[%c0_5] : memref<?xi32>
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:        taskflow.hyperblock.yield
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      }) : (index) -> ()
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      taskflow.yield writes(%arg13 : memref<?xi32>)
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:    }
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:    %dependency_write_out_2 = taskflow.task @Task_3 dependency_read_in(%arg3 : memref<?x7xi32>) dependency_write_in(%arg7 : memref<?xi32>) [original_read_memrefs(%arg3 : memref<?x7xi32>), original_write_memrefs(%arg7 : memref<?xi32>)] {profile_info = {duration = 1 : i32}, task_orchestration_info = {cgra_positions = [{col = 2 : i32, context_id = 0 : i32, row = 0 : i32}], read_sram_locations = [{col = 2 : i32, row = 0 : i32}], write_sram_locations = [{col = 3 : i32, row = 0 : i32}]}} : (memref<?x7xi32>, memref<?xi32>) -> (memref<?xi32>) {
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:    ^bb0(%arg10: memref<?x7xi32>, %arg11: memref<?xi32>):
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %c0 = arith.constant 0 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %c4 = arith.constant 4 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %c1 = arith.constant 1 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %1 = taskflow.counter from %c0 to %c4 step %c1 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %c7 = arith.constant 7 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %2 = taskflow.counter parent(%1 : index) from %c0 to %c7 step %c1 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      "taskflow.hyperblock"(%1, %2) <{operandSegmentSizes = array<i32: 2, 0>}> ({
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      ^bb0(%arg12: index, %arg13: index):
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:        %3 = memref.load %arg10[%arg12, %arg13] : memref<?x7xi32>
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:        memref.store %3, %arg11[%arg13] : memref<?xi32>
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:        taskflow.hyperblock.yield
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      }) : (index, index) -> ()
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      taskflow.yield writes(%arg11 : memref<?xi32>)
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:    }
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:    %dependency_write_out_3 = taskflow.task @Task_4 dependency_read_in(%arg4, %dependency_write_out_2 : memref<?x9xi32>, memref<?xi32>) dependency_write_in(%arg8 : memref<?xi32>) [original_read_memrefs(%arg4, %arg7 : memref<?x9xi32>, memref<?xi32>), original_write_memrefs(%arg8 : memref<?xi32>)] {profile_info = {duration = 1 : i32}, task_orchestration_info = {cgra_positions = [{col = 3 : i32, context_id = 0 : i32, row = 0 : i32}], read_sram_locations = [{col = 3 : i32, row = 0 : i32}, {col = 3 : i32, row = 0 : i32}], write_sram_locations = [{col = 3 : i32, row = 0 : i32}]}} : (memref<?x9xi32>, memref<?xi32>, memref<?xi32>) -> (memref<?xi32>) {
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:    ^bb0(%arg10: memref<?x9xi32>, %arg11: memref<?xi32>, %arg12: memref<?xi32>):
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %c0 = arith.constant 0 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %c4 = arith.constant 4 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %c1 = arith.constant 1 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %1 = taskflow.counter from %c0 to %c4 step %c1 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %c9 = arith.constant 9 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      %2 = taskflow.counter parent(%1 : index) from %c0 to %c9 step %c1 : index
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      "taskflow.hyperblock"(%1, %2) <{operandSegmentSizes = array<i32: 2, 0>}> ({
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      ^bb0(%arg13: index, %arg14: index):
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:        %3 = memref.load %arg10[%arg13, %arg14] : memref<?x9xi32>
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:        %4 = memref.load %arg11[%arg14] : memref<?xi32>
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:        %5 = arith.addi %3, %4 : i32
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:        memref.store %5, %arg12[%arg14] : memref<?xi32>
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:        taskflow.hyperblock.yield
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      }) : (index, index) -> ()
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:      taskflow.yield writes(%arg12 : memref<?xi32>)
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:    }
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:    %0 = affine.load %dependency_write_out_1[0] : memref<?xi32>
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:    return %0 : i32
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:  }
// MAP-SPATIAL-TEMPORAL-4x4-NEXT:}

// MAP-SPATIAL-TEMPORAL-1x1:     module {
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:  func.func @_Z21pureNestedLoopExamplePA8_A6_iPA8_A5_iS4_PA7_iPA9_iPiS9_S9_S9_S9_(%arg0: memref<?x8x6xi32>, %arg1: memref<?x8x5xi32>, %arg2: memref<?x8x5xi32>, %arg3: memref<?x7xi32>, %arg4: memref<?x9xi32>, %arg5: memref<?xi32>, %arg6: memref<?xi32>, %arg7: memref<?xi32>, %arg8: memref<?xi32>, %arg9: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:    %dependency_write_out = taskflow.task @Task_0 dependency_read_in(%arg0 : memref<?x8x6xi32>) dependency_write_in(%arg5 : memref<?xi32>) [original_read_memrefs(%arg0 : memref<?x8x6xi32>), original_write_memrefs(%arg5 : memref<?xi32>)] {profile_info = {duration = 1 : i32}, task_orchestration_info = {cgra_positions = [{col = 0 : i32, context_id = 0 : i32, row = 0 : i32}], read_sram_locations = [{col = 0 : i32, row = 0 : i32}], write_sram_locations = [{col = 0 : i32, row = 0 : i32}]}} : (memref<?x8x6xi32>, memref<?xi32>) -> (memref<?xi32>) {
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:    ^bb0(%arg10: memref<?x8x6xi32>, %arg11: memref<?xi32>):
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %c0 = arith.constant 0 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %c4 = arith.constant 4 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %c1 = arith.constant 1 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %1 = taskflow.counter from %c0 to %c4 step %c1 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %c8 = arith.constant 8 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %2 = taskflow.counter parent(%1 : index) from %c0 to %c8 step %c1 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %c6 = arith.constant 6 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %3 = taskflow.counter parent(%2 : index) from %c0 to %c6 step %c1 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      "taskflow.hyperblock"(%1, %2, %3) <{operandSegmentSizes = array<i32: 3, 0>}> ({
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      ^bb0(%arg12: index, %arg13: index, %arg14: index):
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:        %4 = memref.load %arg10[%arg12, %arg13, %arg14] : memref<?x8x6xi32>
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:        memref.store %4, %arg11[%arg14] : memref<?xi32>
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:        taskflow.hyperblock.yield
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      }) : (index, index, index) -> ()
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      taskflow.yield writes(%arg11 : memref<?xi32>)
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:    }
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:    %dependency_write_out_0 = taskflow.task @Task_1 dependency_read_in(%arg1, %arg2 : memref<?x8x5xi32>, memref<?x8x5xi32>) dependency_write_in(%arg6 : memref<?xi32>) [original_read_memrefs(%arg1, %arg2 : memref<?x8x5xi32>, memref<?x8x5xi32>), original_write_memrefs(%arg6 : memref<?xi32>)] {profile_info = {duration = 1 : i32}, task_orchestration_info = {cgra_positions = [{col = 0 : i32, context_id = 1 : i32, row = 0 : i32}], read_sram_locations = [{col = 0 : i32, row = 0 : i32}, {col = 0 : i32, row = 0 : i32}], write_sram_locations = [{col = 0 : i32, row = 0 : i32}]}} : (memref<?x8x5xi32>, memref<?x8x5xi32>, memref<?xi32>) -> (memref<?xi32>) {
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:    ^bb0(%arg10: memref<?x8x5xi32>, %arg11: memref<?x8x5xi32>, %arg12: memref<?xi32>):
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %c0 = arith.constant 0 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %c4 = arith.constant 4 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %c1 = arith.constant 1 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %1 = taskflow.counter from %c0 to %c4 step %c1 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %c8 = arith.constant 8 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %2 = taskflow.counter parent(%1 : index) from %c0 to %c8 step %c1 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %c5 = arith.constant 5 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %3 = taskflow.counter parent(%2 : index) from %c0 to %c5 step %c1 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      "taskflow.hyperblock"(%1, %2, %3) <{operandSegmentSizes = array<i32: 3, 0>}> ({
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      ^bb0(%arg13: index, %arg14: index, %arg15: index):
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:        %4 = memref.load %arg10[%arg13, %arg14, %arg15] : memref<?x8x5xi32>
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:        %5 = memref.load %arg11[%arg13, %arg14, %arg15] : memref<?x8x5xi32>
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:        %6 = arith.addi %4, %5 : i32
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:        memref.store %6, %arg12[%arg15] : memref<?xi32>
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:        taskflow.hyperblock.yield
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      }) : (index, index, index) -> ()
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      taskflow.yield writes(%arg12 : memref<?xi32>)
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:    }
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:    %dependency_write_out_1 = taskflow.task @Task_2 dependency_read_in(%dependency_write_out, %dependency_write_out_0, %arg9 : memref<?xi32>, memref<?xi32>, memref<?xi32>) dependency_write_in(%arg9 : memref<?xi32>) [original_read_memrefs(%arg5, %arg6, %arg9 : memref<?xi32>, memref<?xi32>, memref<?xi32>), original_write_memrefs(%arg9 : memref<?xi32>)] {profile_info = {duration = 1 : i32}, task_orchestration_info = {cgra_positions = [{col = 0 : i32, context_id = 3 : i32, row = 0 : i32}], read_sram_locations = [{col = 0 : i32, row = 0 : i32}, {col = 0 : i32, row = 0 : i32}, {col = 0 : i32, row = 0 : i32}], write_sram_locations = [{col = 0 : i32, row = 0 : i32}]}} : (memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) -> (memref<?xi32>) {
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:    ^bb0(%arg10: memref<?xi32>, %arg11: memref<?xi32>, %arg12: memref<?xi32>, %arg13: memref<?xi32>):
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %c0 = arith.constant 0 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %c4 = arith.constant 4 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %c1 = arith.constant 1 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %1 = taskflow.counter from %c0 to %c4 step %c1 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %c8 = arith.constant 8 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %2 = taskflow.counter parent(%1 : index) from %c0 to %c8 step %c1 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %c6 = arith.constant 6 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %3 = taskflow.counter parent(%2 : index) from %c0 to %c6 step %c1 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      "taskflow.hyperblock"(%3) <{operandSegmentSizes = array<i32: 1, 0>}> ({
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      ^bb0(%arg14: index):
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:        %4 = memref.load %arg10[%arg14] : memref<?xi32>
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:        %5 = memref.load %arg11[%arg14] : memref<?xi32>
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:        %6 = arith.addi %4, %5 : i32
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:        %c0_4 = arith.constant 0 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:        %7 = memref.load %arg13[%c0_4] : memref<?xi32>
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:        %8 = arith.addi %7, %6 : i32
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:        %c0_5 = arith.constant 0 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:        memref.store %8, %arg13[%c0_5] : memref<?xi32>
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:        taskflow.hyperblock.yield
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      }) : (index) -> ()
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      taskflow.yield writes(%arg13 : memref<?xi32>)
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:    }
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:    %dependency_write_out_2 = taskflow.task @Task_3 dependency_read_in(%arg3 : memref<?x7xi32>) dependency_write_in(%arg7 : memref<?xi32>) [original_read_memrefs(%arg3 : memref<?x7xi32>), original_write_memrefs(%arg7 : memref<?xi32>)] {profile_info = {duration = 1 : i32}, task_orchestration_info = {cgra_positions = [{col = 0 : i32, context_id = 2 : i32, row = 0 : i32}], read_sram_locations = [{col = 0 : i32, row = 0 : i32}], write_sram_locations = [{col = 0 : i32, row = 0 : i32}]}} : (memref<?x7xi32>, memref<?xi32>) -> (memref<?xi32>) {
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:    ^bb0(%arg10: memref<?x7xi32>, %arg11: memref<?xi32>):
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %c0 = arith.constant 0 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %c4 = arith.constant 4 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %c1 = arith.constant 1 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %1 = taskflow.counter from %c0 to %c4 step %c1 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %c7 = arith.constant 7 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %2 = taskflow.counter parent(%1 : index) from %c0 to %c7 step %c1 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      "taskflow.hyperblock"(%1, %2) <{operandSegmentSizes = array<i32: 2, 0>}> ({
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      ^bb0(%arg12: index, %arg13: index):
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:        %3 = memref.load %arg10[%arg12, %arg13] : memref<?x7xi32>
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:        memref.store %3, %arg11[%arg13] : memref<?xi32>
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:        taskflow.hyperblock.yield
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      }) : (index, index) -> ()
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      taskflow.yield writes(%arg11 : memref<?xi32>)
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:    }
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:    %dependency_write_out_3 = taskflow.task @Task_4 dependency_read_in(%arg4, %dependency_write_out_2 : memref<?x9xi32>, memref<?xi32>) dependency_write_in(%arg8 : memref<?xi32>) [original_read_memrefs(%arg4, %arg7 : memref<?x9xi32>, memref<?xi32>), original_write_memrefs(%arg8 : memref<?xi32>)] {profile_info = {duration = 1 : i32}, task_orchestration_info = {cgra_positions = [{col = 0 : i32, context_id = 4 : i32, row = 0 : i32}], read_sram_locations = [{col = 0 : i32, row = 0 : i32}, {col = 0 : i32, row = 0 : i32}], write_sram_locations = [{col = 0 : i32, row = 0 : i32}]}} : (memref<?x9xi32>, memref<?xi32>, memref<?xi32>) -> (memref<?xi32>) {
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:    ^bb0(%arg10: memref<?x9xi32>, %arg11: memref<?xi32>, %arg12: memref<?xi32>):
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %c0 = arith.constant 0 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %c4 = arith.constant 4 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %c1 = arith.constant 1 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %1 = taskflow.counter from %c0 to %c4 step %c1 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %c9 = arith.constant 9 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      %2 = taskflow.counter parent(%1 : index) from %c0 to %c9 step %c1 : index
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      "taskflow.hyperblock"(%1, %2) <{operandSegmentSizes = array<i32: 2, 0>}> ({
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      ^bb0(%arg13: index, %arg14: index):
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:        %3 = memref.load %arg10[%arg13, %arg14] : memref<?x9xi32>
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:        %4 = memref.load %arg11[%arg14] : memref<?xi32>
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:        %5 = arith.addi %3, %4 : i32
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:        memref.store %5, %arg12[%arg14] : memref<?xi32>
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:        taskflow.hyperblock.yield
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      }) : (index, index) -> ()
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:      taskflow.yield writes(%arg12 : memref<?xi32>)
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:    }
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:    %0 = affine.load %dependency_write_out_1[0] : memref<?xi32>
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:    return %0 : i32
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:  }
// MAP-SPATIAL-TEMPORAL-1x1-NEXT:}

// MAP-SPATIAL-TEMPORAL-1x2:     module {
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:  func.func @_Z21pureNestedLoopExamplePA8_A6_iPA8_A5_iS4_PA7_iPA9_iPiS9_S9_S9_S9_(%arg0: memref<?x8x6xi32>, %arg1: memref<?x8x5xi32>, %arg2: memref<?x8x5xi32>, %arg3: memref<?x7xi32>, %arg4: memref<?x9xi32>, %arg5: memref<?xi32>, %arg6: memref<?xi32>, %arg7: memref<?xi32>, %arg8: memref<?xi32>, %arg9: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:    %dependency_write_out = taskflow.task @Task_0 dependency_read_in(%arg0 : memref<?x8x6xi32>) dependency_write_in(%arg5 : memref<?xi32>) [original_read_memrefs(%arg0 : memref<?x8x6xi32>), original_write_memrefs(%arg5 : memref<?xi32>)] {profile_info = {duration = 1 : i32}, task_orchestration_info = {cgra_positions = [{col = 0 : i32, context_id = 0 : i32, row = 0 : i32}], read_sram_locations = [{col = 0 : i32, row = 0 : i32}], write_sram_locations = [{col = 1 : i32, row = 0 : i32}]}} : (memref<?x8x6xi32>, memref<?xi32>) -> (memref<?xi32>) {
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:    ^bb0(%arg10: memref<?x8x6xi32>, %arg11: memref<?xi32>):
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %c0 = arith.constant 0 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %c4 = arith.constant 4 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %c1 = arith.constant 1 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %1 = taskflow.counter from %c0 to %c4 step %c1 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %c8 = arith.constant 8 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %2 = taskflow.counter parent(%1 : index) from %c0 to %c8 step %c1 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %c6 = arith.constant 6 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %3 = taskflow.counter parent(%2 : index) from %c0 to %c6 step %c1 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      "taskflow.hyperblock"(%1, %2, %3) <{operandSegmentSizes = array<i32: 3, 0>}> ({
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      ^bb0(%arg12: index, %arg13: index, %arg14: index):
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:        %4 = memref.load %arg10[%arg12, %arg13, %arg14] : memref<?x8x6xi32>
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:        memref.store %4, %arg11[%arg14] : memref<?xi32>
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:        taskflow.hyperblock.yield
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      }) : (index, index, index) -> ()
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      taskflow.yield writes(%arg11 : memref<?xi32>)
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:    }
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:    %dependency_write_out_0 = taskflow.task @Task_1 dependency_read_in(%arg1, %arg2 : memref<?x8x5xi32>, memref<?x8x5xi32>) dependency_write_in(%arg6 : memref<?xi32>) [original_read_memrefs(%arg1, %arg2 : memref<?x8x5xi32>, memref<?x8x5xi32>), original_write_memrefs(%arg6 : memref<?xi32>)] {profile_info = {duration = 1 : i32}, task_orchestration_info = {cgra_positions = [{col = 1 : i32, context_id = 0 : i32, row = 0 : i32}], read_sram_locations = [{col = 1 : i32, row = 0 : i32}, {col = 1 : i32, row = 0 : i32}], write_sram_locations = [{col = 1 : i32, row = 0 : i32}]}} : (memref<?x8x5xi32>, memref<?x8x5xi32>, memref<?xi32>) -> (memref<?xi32>) {
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:    ^bb0(%arg10: memref<?x8x5xi32>, %arg11: memref<?x8x5xi32>, %arg12: memref<?xi32>):
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %c0 = arith.constant 0 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %c4 = arith.constant 4 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %c1 = arith.constant 1 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %1 = taskflow.counter from %c0 to %c4 step %c1 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %c8 = arith.constant 8 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %2 = taskflow.counter parent(%1 : index) from %c0 to %c8 step %c1 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %c5 = arith.constant 5 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %3 = taskflow.counter parent(%2 : index) from %c0 to %c5 step %c1 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      "taskflow.hyperblock"(%1, %2, %3) <{operandSegmentSizes = array<i32: 3, 0>}> ({
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      ^bb0(%arg13: index, %arg14: index, %arg15: index):
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:        %4 = memref.load %arg10[%arg13, %arg14, %arg15] : memref<?x8x5xi32>
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:        %5 = memref.load %arg11[%arg13, %arg14, %arg15] : memref<?x8x5xi32>
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:        %6 = arith.addi %4, %5 : i32
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:        memref.store %6, %arg12[%arg15] : memref<?xi32>
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:        taskflow.hyperblock.yield
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      }) : (index, index, index) -> ()
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      taskflow.yield writes(%arg12 : memref<?xi32>)
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:    }
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:    %dependency_write_out_1 = taskflow.task @Task_2 dependency_read_in(%dependency_write_out, %dependency_write_out_0, %arg9 : memref<?xi32>, memref<?xi32>, memref<?xi32>) dependency_write_in(%arg9 : memref<?xi32>) [original_read_memrefs(%arg5, %arg6, %arg9 : memref<?xi32>, memref<?xi32>, memref<?xi32>), original_write_memrefs(%arg9 : memref<?xi32>)] {profile_info = {duration = 1 : i32}, task_orchestration_info = {cgra_positions = [{col = 1 : i32, context_id = 1 : i32, row = 0 : i32}], read_sram_locations = [{col = 1 : i32, row = 0 : i32}, {col = 1 : i32, row = 0 : i32}, {col = 1 : i32, row = 0 : i32}], write_sram_locations = [{col = 1 : i32, row = 0 : i32}]}} : (memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) -> (memref<?xi32>) {
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:    ^bb0(%arg10: memref<?xi32>, %arg11: memref<?xi32>, %arg12: memref<?xi32>, %arg13: memref<?xi32>):
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %c0 = arith.constant 0 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %c4 = arith.constant 4 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %c1 = arith.constant 1 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %1 = taskflow.counter from %c0 to %c4 step %c1 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %c8 = arith.constant 8 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %2 = taskflow.counter parent(%1 : index) from %c0 to %c8 step %c1 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %c6 = arith.constant 6 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %3 = taskflow.counter parent(%2 : index) from %c0 to %c6 step %c1 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      "taskflow.hyperblock"(%3) <{operandSegmentSizes = array<i32: 1, 0>}> ({
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      ^bb0(%arg14: index):
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:        %4 = memref.load %arg10[%arg14] : memref<?xi32>
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:        %5 = memref.load %arg11[%arg14] : memref<?xi32>
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:        %6 = arith.addi %4, %5 : i32
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:        %c0_4 = arith.constant 0 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:        %7 = memref.load %arg13[%c0_4] : memref<?xi32>
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:        %8 = arith.addi %7, %6 : i32
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:        %c0_5 = arith.constant 0 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:        memref.store %8, %arg13[%c0_5] : memref<?xi32>
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:        taskflow.hyperblock.yield
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      }) : (index) -> ()
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      taskflow.yield writes(%arg13 : memref<?xi32>)
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:    }
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:    %dependency_write_out_2 = taskflow.task @Task_3 dependency_read_in(%arg3 : memref<?x7xi32>) dependency_write_in(%arg7 : memref<?xi32>) [original_read_memrefs(%arg3 : memref<?x7xi32>), original_write_memrefs(%arg7 : memref<?xi32>)] {profile_info = {duration = 1 : i32}, task_orchestration_info = {cgra_positions = [{col = 0 : i32, context_id = 1 : i32, row = 0 : i32}], read_sram_locations = [{col = 0 : i32, row = 0 : i32}], write_sram_locations = [{col = 0 : i32, row = 0 : i32}]}} : (memref<?x7xi32>, memref<?xi32>) -> (memref<?xi32>) {
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:    ^bb0(%arg10: memref<?x7xi32>, %arg11: memref<?xi32>):
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %c0 = arith.constant 0 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %c4 = arith.constant 4 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %c1 = arith.constant 1 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %1 = taskflow.counter from %c0 to %c4 step %c1 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %c7 = arith.constant 7 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %2 = taskflow.counter parent(%1 : index) from %c0 to %c7 step %c1 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      "taskflow.hyperblock"(%1, %2) <{operandSegmentSizes = array<i32: 2, 0>}> ({
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      ^bb0(%arg12: index, %arg13: index):
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:        %3 = memref.load %arg10[%arg12, %arg13] : memref<?x7xi32>
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:        memref.store %3, %arg11[%arg13] : memref<?xi32>
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:        taskflow.hyperblock.yield
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      }) : (index, index) -> ()
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      taskflow.yield writes(%arg11 : memref<?xi32>)
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:    }
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:    %dependency_write_out_3 = taskflow.task @Task_4 dependency_read_in(%arg4, %dependency_write_out_2 : memref<?x9xi32>, memref<?xi32>) dependency_write_in(%arg8 : memref<?xi32>) [original_read_memrefs(%arg4, %arg7 : memref<?x9xi32>, memref<?xi32>), original_write_memrefs(%arg8 : memref<?xi32>)] {profile_info = {duration = 1 : i32}, task_orchestration_info = {cgra_positions = [{col = 0 : i32, context_id = 2 : i32, row = 0 : i32}], read_sram_locations = [{col = 0 : i32, row = 0 : i32}, {col = 0 : i32, row = 0 : i32}], write_sram_locations = [{col = 0 : i32, row = 0 : i32}]}} : (memref<?x9xi32>, memref<?xi32>, memref<?xi32>) -> (memref<?xi32>) {
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:    ^bb0(%arg10: memref<?x9xi32>, %arg11: memref<?xi32>, %arg12: memref<?xi32>):
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %c0 = arith.constant 0 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %c4 = arith.constant 4 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %c1 = arith.constant 1 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %1 = taskflow.counter from %c0 to %c4 step %c1 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %c9 = arith.constant 9 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      %2 = taskflow.counter parent(%1 : index) from %c0 to %c9 step %c1 : index
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      "taskflow.hyperblock"(%1, %2) <{operandSegmentSizes = array<i32: 2, 0>}> ({
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      ^bb0(%arg13: index, %arg14: index):
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:        %3 = memref.load %arg10[%arg13, %arg14] : memref<?x9xi32>
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:        %4 = memref.load %arg11[%arg14] : memref<?xi32>
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:        %5 = arith.addi %3, %4 : i32
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:        memref.store %5, %arg12[%arg14] : memref<?xi32>
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:        taskflow.hyperblock.yield
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      }) : (index, index) -> ()
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:      taskflow.yield writes(%arg12 : memref<?xi32>)
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:    }
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:    %0 = affine.load %dependency_write_out_1[0] : memref<?xi32>
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:    return %0 : i32
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:  }
// MAP-SPATIAL-TEMPORAL-1x2-NEXT:}

// MAP-SPATIAL:      module {
// MAP-SPATIAL-NEXT:   func.func @_Z21pureNestedLoopExamplePA8_A6_iPA8_A5_iS4_PA7_iPA9_iPiS9_S9_S9_S9_(%arg0: memref<?x8x6xi32>, %arg1: memref<?x8x5xi32>, %arg2: memref<?x8x5xi32>, %arg3: memref<?x7xi32>, %arg4: memref<?x9xi32>, %arg5: memref<?xi32>, %arg6: memref<?xi32>, %arg7: memref<?xi32>, %arg8: memref<?xi32>, %arg9: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// MAP-SPATIAL-NEXT:     %dependency_write_out = taskflow.task @Task_0 dependency_read_in(%arg0 : memref<?x8x6xi32>) dependency_write_in(%arg5 : memref<?xi32>) [original_read_memrefs(%arg0 : memref<?x8x6xi32>), original_write_memrefs(%arg5 : memref<?xi32>)] {profile_info = {duration = 1 : i32}, task_orchestration_info = {cgra_positions = [{col = 0 : i32, context_id = 0 : i32, row = 0 : i32}], read_sram_locations = [{col = 0 : i32, row = 0 : i32}], write_sram_locations = [{col = 0 : i32, row = 1 : i32}]}} : (memref<?x8x6xi32>, memref<?xi32>) -> (memref<?xi32>) {
// MAP-SPATIAL-NEXT:     ^bb0(%arg10: memref<?x8x6xi32>, %arg11: memref<?xi32>):
// MAP-SPATIAL-NEXT:       %c0 = arith.constant 0 : index
// MAP-SPATIAL-NEXT:       %c4 = arith.constant 4 : index
// MAP-SPATIAL-NEXT:       %c1 = arith.constant 1 : index
// MAP-SPATIAL-NEXT:       %1 = taskflow.counter from %c0 to %c4 step %c1 : index
// MAP-SPATIAL-NEXT:       %c8 = arith.constant 8 : index
// MAP-SPATIAL-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0 to %c8 step %c1 : index
// MAP-SPATIAL-NEXT:       %c6 = arith.constant 6 : index
// MAP-SPATIAL-NEXT:       %3 = taskflow.counter parent(%2 : index) from %c0 to %c6 step %c1 : index
// MAP-SPATIAL-NEXT:       "taskflow.hyperblock"(%1, %2, %3) <{operandSegmentSizes = array<i32: 3, 0>}> ({
// MAP-SPATIAL-NEXT:       ^bb0(%arg12: index, %arg13: index, %arg14: index):
// MAP-SPATIAL-NEXT:         %4 = memref.load %arg10[%arg12, %arg13, %arg14] : memref<?x8x6xi32>
// MAP-SPATIAL-NEXT:         memref.store %4, %arg11[%arg14] : memref<?xi32>
// MAP-SPATIAL-NEXT:         taskflow.hyperblock.yield
// MAP-SPATIAL-NEXT:       }) : (index, index, index) -> ()
// MAP-SPATIAL-NEXT:       taskflow.yield writes(%arg11 : memref<?xi32>)
// MAP-SPATIAL-NEXT:     }
// MAP-SPATIAL-NEXT:     %dependency_write_out_0 = taskflow.task @Task_1 dependency_read_in(%arg1, %arg2 : memref<?x8x5xi32>, memref<?x8x5xi32>) dependency_write_in(%arg6 : memref<?xi32>) [original_read_memrefs(%arg1, %arg2 : memref<?x8x5xi32>, memref<?x8x5xi32>), original_write_memrefs(%arg6 : memref<?xi32>)] {profile_info = {duration = 1 : i32}, task_orchestration_info = {cgra_positions = [{col = 1 : i32, context_id = 0 : i32, row = 0 : i32}], read_sram_locations = [{col = 1 : i32, row = 0 : i32}, {col = 1 : i32, row = 0 : i32}], write_sram_locations = [{col = 1 : i32, row = 1 : i32}]}} : (memref<?x8x5xi32>, memref<?x8x5xi32>, memref<?xi32>) -> (memref<?xi32>) {
// MAP-SPATIAL-NEXT:     ^bb0(%arg10: memref<?x8x5xi32>, %arg11: memref<?x8x5xi32>, %arg12: memref<?xi32>):
// MAP-SPATIAL-NEXT:       %c0 = arith.constant 0 : index
// MAP-SPATIAL-NEXT:       %c4 = arith.constant 4 : index
// MAP-SPATIAL-NEXT:       %c1 = arith.constant 1 : index
// MAP-SPATIAL-NEXT:       %1 = taskflow.counter from %c0 to %c4 step %c1 : index
// MAP-SPATIAL-NEXT:       %c8 = arith.constant 8 : index
// MAP-SPATIAL-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0 to %c8 step %c1 : index
// MAP-SPATIAL-NEXT:       %c5 = arith.constant 5 : index
// MAP-SPATIAL-NEXT:       %3 = taskflow.counter parent(%2 : index) from %c0 to %c5 step %c1 : index
// MAP-SPATIAL-NEXT:       "taskflow.hyperblock"(%1, %2, %3) <{operandSegmentSizes = array<i32: 3, 0>}> ({
// MAP-SPATIAL-NEXT:       ^bb0(%arg13: index, %arg14: index, %arg15: index):
// MAP-SPATIAL-NEXT:         %4 = memref.load %arg10[%arg13, %arg14, %arg15] : memref<?x8x5xi32>
// MAP-SPATIAL-NEXT:         %5 = memref.load %arg11[%arg13, %arg14, %arg15] : memref<?x8x5xi32>
// MAP-SPATIAL-NEXT:         %6 = arith.addi %4, %5 : i32
// MAP-SPATIAL-NEXT:         memref.store %6, %arg12[%arg15] : memref<?xi32>
// MAP-SPATIAL-NEXT:         taskflow.hyperblock.yield
// MAP-SPATIAL-NEXT:       }) : (index, index, index) -> ()
// MAP-SPATIAL-NEXT:       taskflow.yield writes(%arg12 : memref<?xi32>)
// MAP-SPATIAL-NEXT:     }
// MAP-SPATIAL-NEXT:     %dependency_write_out_1 = taskflow.task @Task_2 dependency_read_in(%dependency_write_out, %dependency_write_out_0, %arg9 : memref<?xi32>, memref<?xi32>, memref<?xi32>) dependency_write_in(%arg9 : memref<?xi32>) [original_read_memrefs(%arg5, %arg6, %arg9 : memref<?xi32>, memref<?xi32>, memref<?xi32>), original_write_memrefs(%arg9 : memref<?xi32>)] {profile_info = {duration = 1 : i32}, task_orchestration_info = {cgra_positions = [{col = 0 : i32, context_id = 0 : i32, row = 1 : i32}], read_sram_locations = [{col = 0 : i32, row = 1 : i32}, {col = 1 : i32, row = 1 : i32}, {col = 0 : i32, row = 1 : i32}], write_sram_locations = [{col = 0 : i32, row = 1 : i32}]}} : (memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) -> (memref<?xi32>) {
// MAP-SPATIAL-NEXT:     ^bb0(%arg10: memref<?xi32>, %arg11: memref<?xi32>, %arg12: memref<?xi32>, %arg13: memref<?xi32>):
// MAP-SPATIAL-NEXT:       %c0 = arith.constant 0 : index
// MAP-SPATIAL-NEXT:       %c4 = arith.constant 4 : index
// MAP-SPATIAL-NEXT:       %c1 = arith.constant 1 : index
// MAP-SPATIAL-NEXT:       %1 = taskflow.counter from %c0 to %c4 step %c1 : index
// MAP-SPATIAL-NEXT:       %c8 = arith.constant 8 : index
// MAP-SPATIAL-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0 to %c8 step %c1 : index
// MAP-SPATIAL-NEXT:       %c6 = arith.constant 6 : index
// MAP-SPATIAL-NEXT:       %3 = taskflow.counter parent(%2 : index) from %c0 to %c6 step %c1 : index
// MAP-SPATIAL-NEXT:       "taskflow.hyperblock"(%3) <{operandSegmentSizes = array<i32: 1, 0>}> ({
// MAP-SPATIAL-NEXT:       ^bb0(%arg14: index):
// MAP-SPATIAL-NEXT:         %4 = memref.load %arg10[%arg14] : memref<?xi32>
// MAP-SPATIAL-NEXT:         %5 = memref.load %arg11[%arg14] : memref<?xi32>
// MAP-SPATIAL-NEXT:         %6 = arith.addi %4, %5 : i32
// MAP-SPATIAL-NEXT:         %c0_4 = arith.constant 0 : index
// MAP-SPATIAL-NEXT:         %7 = memref.load %arg13[%c0_4] : memref<?xi32>
// MAP-SPATIAL-NEXT:         %8 = arith.addi %7, %6 : i32
// MAP-SPATIAL-NEXT:         %c0_5 = arith.constant 0 : index
// MAP-SPATIAL-NEXT:         memref.store %8, %arg13[%c0_5] : memref<?xi32>
// MAP-SPATIAL-NEXT:         taskflow.hyperblock.yield
// MAP-SPATIAL-NEXT:       }) : (index) -> ()
// MAP-SPATIAL-NEXT:       taskflow.yield writes(%arg13 : memref<?xi32>)
// MAP-SPATIAL-NEXT:     }
// MAP-SPATIAL-NEXT:     %dependency_write_out_2 = taskflow.task @Task_3 dependency_read_in(%arg3 : memref<?x7xi32>) dependency_write_in(%arg7 : memref<?xi32>) [original_read_memrefs(%arg3 : memref<?x7xi32>), original_write_memrefs(%arg7 : memref<?xi32>)] {profile_info = {duration = 1 : i32}, task_orchestration_info = {cgra_positions = [{col = 2 : i32, context_id = 0 : i32, row = 0 : i32}], read_sram_locations = [{col = 2 : i32, row = 0 : i32}], write_sram_locations = [{col = 3 : i32, row = 0 : i32}]}} : (memref<?x7xi32>, memref<?xi32>) -> (memref<?xi32>) {
// MAP-SPATIAL-NEXT:     ^bb0(%arg10: memref<?x7xi32>, %arg11: memref<?xi32>):
// MAP-SPATIAL-NEXT:       %c0 = arith.constant 0 : index
// MAP-SPATIAL-NEXT:       %c4 = arith.constant 4 : index
// MAP-SPATIAL-NEXT:       %c1 = arith.constant 1 : index
// MAP-SPATIAL-NEXT:       %1 = taskflow.counter from %c0 to %c4 step %c1 : index
// MAP-SPATIAL-NEXT:       %c7 = arith.constant 7 : index
// MAP-SPATIAL-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0 to %c7 step %c1 : index
// MAP-SPATIAL-NEXT:       "taskflow.hyperblock"(%1, %2) <{operandSegmentSizes = array<i32: 2, 0>}> ({
// MAP-SPATIAL-NEXT:       ^bb0(%arg12: index, %arg13: index):
// MAP-SPATIAL-NEXT:         %3 = memref.load %arg10[%arg12, %arg13] : memref<?x7xi32>
// MAP-SPATIAL-NEXT:         memref.store %3, %arg11[%arg13] : memref<?xi32>
// MAP-SPATIAL-NEXT:         taskflow.hyperblock.yield
// MAP-SPATIAL-NEXT:       }) : (index, index) -> ()
// MAP-SPATIAL-NEXT:       taskflow.yield writes(%arg11 : memref<?xi32>)
// MAP-SPATIAL-NEXT:     }
// MAP-SPATIAL-NEXT:     %dependency_write_out_3 = taskflow.task @Task_4 dependency_read_in(%arg4, %dependency_write_out_2 : memref<?x9xi32>, memref<?xi32>) dependency_write_in(%arg8 : memref<?xi32>) [original_read_memrefs(%arg4, %arg7 : memref<?x9xi32>, memref<?xi32>), original_write_memrefs(%arg8 : memref<?xi32>)] {profile_info = {duration = 1 : i32}, task_orchestration_info = {cgra_positions = [{col = 3 : i32, context_id = 0 : i32, row = 0 : i32}], read_sram_locations = [{col = 3 : i32, row = 0 : i32}, {col = 3 : i32, row = 0 : i32}], write_sram_locations = [{col = 3 : i32, row = 0 : i32}]}} : (memref<?x9xi32>, memref<?xi32>, memref<?xi32>) -> (memref<?xi32>) {
// MAP-SPATIAL-NEXT:     ^bb0(%arg10: memref<?x9xi32>, %arg11: memref<?xi32>, %arg12: memref<?xi32>):
// MAP-SPATIAL-NEXT:       %c0 = arith.constant 0 : index
// MAP-SPATIAL-NEXT:       %c4 = arith.constant 4 : index
// MAP-SPATIAL-NEXT:       %c1 = arith.constant 1 : index
// MAP-SPATIAL-NEXT:       %1 = taskflow.counter from %c0 to %c4 step %c1 : index
// MAP-SPATIAL-NEXT:       %c9 = arith.constant 9 : index
// MAP-SPATIAL-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0 to %c9 step %c1 : index
// MAP-SPATIAL-NEXT:       "taskflow.hyperblock"(%1, %2) <{operandSegmentSizes = array<i32: 2, 0>}> ({
// MAP-SPATIAL-NEXT:       ^bb0(%arg13: index, %arg14: index):
// MAP-SPATIAL-NEXT:         %3 = memref.load %arg10[%arg13, %arg14] : memref<?x9xi32>
// MAP-SPATIAL-NEXT:         %4 = memref.load %arg11[%arg14] : memref<?xi32>
// MAP-SPATIAL-NEXT:         %5 = arith.addi %3, %4 : i32
// MAP-SPATIAL-NEXT:         memref.store %5, %arg12[%arg14] : memref<?xi32>
// MAP-SPATIAL-NEXT:         taskflow.hyperblock.yield
// MAP-SPATIAL-NEXT:       }) : (index, index) -> ()
// MAP-SPATIAL-NEXT:       taskflow.yield writes(%arg12 : memref<?xi32>)
// MAP-SPATIAL-NEXT:     }
// MAP-SPATIAL-NEXT:     %0 = affine.load %dependency_write_out_1[0] : memref<?xi32>
// MAP-SPATIAL-NEXT:     return %0 : i32
// MAP-SPATIAL-NEXT:   }
// MAP-SPATIAL-NEXT: }


// RESOPT:      module {
// RESOPT-NEXT:   func.func @_Z21pureNestedLoopExamplePA8_A6_iPA8_A5_iS4_PA7_iPA9_iPiS9_S9_S9_S9_(%arg0: memref<?x8x6xi32>, %arg1: memref<?x8x5xi32>, %arg2: memref<?x8x5xi32>, %arg3: memref<?x7xi32>, %arg4: memref<?x9xi32>, %arg5: memref<?xi32>, %arg6: memref<?xi32>, %arg7: memref<?xi32>, %arg8: memref<?xi32>, %arg9: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// RESOPT-NEXT:     %c0 = arith.constant 0 : index
// RESOPT-NEXT:     %dependency_write_out = taskflow.task @Task_1 dependency_read_in(%arg1, %arg2 : memref<?x8x5xi32>, memref<?x8x5xi32>) dependency_write_in(%arg6 : memref<?xi32>) [original_read_memrefs(%arg1, %arg2 : memref<?x8x5xi32>, memref<?x8x5xi32>), original_write_memrefs(%arg6 : memref<?xi32>)] {cgra_count = 2 : i32, cgra_shape = "1x2", compiled_ii = 2 : i32, dlp_replicable = true, profile_info = {duration = 4 : i32}, trip_count = 160 : i32} : (memref<?x8x5xi32>, memref<?x8x5xi32>, memref<?xi32>) -> (memref<?xi32>) {
// RESOPT-NEXT:     ^bb0(%arg10: memref<?x8x5xi32>, %arg11: memref<?x8x5xi32>, %arg12: memref<?xi32>):
// RESOPT-NEXT:       %c5 = arith.constant 5 : index
// RESOPT-NEXT:       %c8 = arith.constant 8 : index
// RESOPT-NEXT:       %c0_2 = arith.constant 0 : index
// RESOPT-NEXT:       %c4 = arith.constant 4 : index
// RESOPT-NEXT:       %c1 = arith.constant 1 : index
// RESOPT-NEXT:       %1 = taskflow.counter from %c0_2 to %c4 step %c1 attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} : index
// RESOPT-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0_2 to %c8 step %c1 attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} : index
// RESOPT-NEXT:       %3 = taskflow.counter parent(%2 : index) from %c0_2 to %c5 step %c1 attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32} : index
// RESOPT-NEXT:       neura.kernel inputs(%arg10, %arg11, %arg12 : memref<?x8x5xi32>, memref<?x8x5xi32>, memref<?xi32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// RESOPT-NEXT:       ^bb0(%arg13: memref<?x8x5xi32>, %arg14: memref<?x8x5xi32>, %arg15: memref<?xi32>):
// RESOPT-NEXT:         %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 4 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %5 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32, lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 8 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %6 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32, lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 5 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %7 = neura.load_indexed [%4, %5, %6 : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>]  {lhs_value = "%input0"} : !neura.data<i32, i1>
// RESOPT-NEXT:         %8 = neura.load_indexed [%4, %5, %6 : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>]  {lhs_value = "%input1"} : !neura.data<i32, i1>
// RESOPT-NEXT:         %9 = "neura.add"(%7, %8) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// RESOPT-NEXT:         neura.store_indexed %9 to [%6 : !neura.data<index, i1>]  {rhs_value = "%input2"} : !neura.data<i32, i1>
// RESOPT-NEXT:         neura.yield {yield_type = "void"}
// RESOPT-NEXT:       }
// RESOPT-NEXT:       taskflow.yield writes(%arg12 : memref<?xi32>)
// RESOPT-NEXT:     }
// RESOPT-NEXT:     %dependency_read_out:4, %dependency_write_out_0:2 = taskflow.task @Task_0_Task_2_fused_Task_3_utilfused dependency_read_in(%arg0, %dependency_write_out, %arg9, %arg3 : memref<?x8x6xi32>, memref<?xi32>, memref<?xi32>, memref<?x7xi32>) dependency_write_in(%arg9, %arg7 : memref<?xi32>, memref<?xi32>) [original_read_memrefs(%arg0, %arg6, %arg9, %arg3 : memref<?x8x6xi32>, memref<?xi32>, memref<?xi32>, memref<?x7xi32>), original_write_memrefs(%arg9, %arg7 : memref<?xi32>, memref<?xi32>)] {cgra_count = 2 : i32, cgra_shape = "1x2", compiled_ii = 2 : i32, profile_info = {duration = 5 : i32}, trip_count = 192 : i32} : (memref<?x8x6xi32>, memref<?xi32>, memref<?xi32>, memref<?x7xi32>, memref<?xi32>, memref<?xi32>) -> (memref<?x8x6xi32>, memref<?xi32>, memref<?xi32>, memref<?x7xi32>, memref<?xi32>, memref<?xi32>) {
// RESOPT-NEXT:     ^bb0(%arg10: memref<?x8x6xi32>, %arg11: memref<?xi32>, %arg12: memref<?xi32>, %arg13: memref<?x7xi32>, %arg14: memref<?xi32>, %arg15: memref<?xi32>):
// RESOPT-NEXT:       %c6 = arith.constant 6 : index
// RESOPT-NEXT:       %c8 = arith.constant 8 : index
// RESOPT-NEXT:       %c0_2 = arith.constant 0 : index
// RESOPT-NEXT:       %c4 = arith.constant 4 : index
// RESOPT-NEXT:       %c1 = arith.constant 1 : index
// RESOPT-NEXT:       %1 = taskflow.counter from %c0_2 to %c4 step %c1 attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} : index
// RESOPT-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0_2 to %c8 step %c1 attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} : index
// RESOPT-NEXT:       %3 = taskflow.counter parent(%2 : index) from %c0_2 to %c6 step %c1 attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32} : index
// RESOPT-NEXT:       neura.kernel inputs(%arg10, %arg11, %arg12, %arg13, %arg15 : memref<?x8x6xi32>, memref<?xi32>, memref<?xi32>, memref<?x7xi32>, memref<?xi32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// RESOPT-NEXT:       ^bb0(%arg16: memref<?x8x6xi32>, %arg17: memref<?xi32>, %arg18: memref<?xi32>, %arg19: memref<?x7xi32>, %arg20: memref<?xi32>):
// RESOPT-NEXT:         %6 = "neura.constant"() <{value = 0 : index}> : () -> !neura.data<index, i1>
// RESOPT-NEXT:         %7 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 4 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %8 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32, lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 8 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %9 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32, lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 6 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %10 = neura.load_indexed [%7, %8, %9 : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>]  {lhs_value = "%input0"} : !neura.data<i32, i1>
// RESOPT-NEXT:         %11 = neura.load_indexed [%9 : !neura.data<index, i1>]  {lhs_value = "%input1"} : !neura.data<i32, i1>
// RESOPT-NEXT:         %12 = "neura.add"(%10, %11) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// RESOPT-NEXT:         %13 = neura.load_indexed [%6 : !neura.data<index, i1>]  {lhs_value = "%input2"} : !neura.data<i32, i1>
// RESOPT-NEXT:         %14 = "neura.add"(%13, %12) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// RESOPT-NEXT:         neura.store_indexed %14 to [%6 : !neura.data<index, i1>]  {rhs_value = "%input2"} : !neura.data<i32, i1>
// RESOPT-NEXT:         %15 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 4 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %16 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 7 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %17 = neura.load_indexed [%15, %16 : !neura.data<index, i1>, !neura.data<index, i1>]  {lhs_value = "%input0"} : !neura.data<i32, i1>
// RESOPT-NEXT:         neura.store_indexed %17 to [%16 : !neura.data<index, i1>]  {rhs_value = "%input1"} : !neura.data<i32, i1>
// RESOPT-NEXT:         neura.yield {yield_type = "void"}
// RESOPT-NEXT:       }
// RESOPT-NEXT:       %c7 = arith.constant 7 : index
// RESOPT-NEXT:       %c0_3 = arith.constant 0 : index
// RESOPT-NEXT:       %c4_4 = arith.constant 4 : index
// RESOPT-NEXT:       %c1_5 = arith.constant 1 : index
// RESOPT-NEXT:       %4 = taskflow.counter from %c0_3 to %c4_4 step %c1_5 attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} : index
// RESOPT-NEXT:       %5 = taskflow.counter parent(%4 : index) from %c0_3 to %c7 step %c1_5 attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} : index
// RESOPT-NEXT:       taskflow.yield reads(%arg10, %arg11, %arg12, %arg13 : memref<?x8x6xi32>, memref<?xi32>, memref<?xi32>, memref<?x7xi32>) writes(%arg14, %arg15 : memref<?xi32>, memref<?xi32>)
// RESOPT-NEXT:     }
// RESOPT-NEXT:     %dependency_write_out_1 = taskflow.task @Task_4 dependency_read_in(%arg4, %dependency_write_out_0#1 : memref<?x9xi32>, memref<?xi32>) dependency_write_in(%arg8 : memref<?xi32>) [original_read_memrefs(%arg4, %arg7 : memref<?x9xi32>, memref<?xi32>), original_write_memrefs(%arg8 : memref<?xi32>)] {cgra_count = 2 : i32, cgra_shape = "1x2", compiled_ii = 2 : i32, dlp_replicable = true, profile_info = {duration = 4 : i32}, trip_count = 36 : i32} : (memref<?x9xi32>, memref<?xi32>, memref<?xi32>) -> (memref<?xi32>) {
// RESOPT-NEXT:     ^bb0(%arg10: memref<?x9xi32>, %arg11: memref<?xi32>, %arg12: memref<?xi32>):
// RESOPT-NEXT:       %c9 = arith.constant 9 : index
// RESOPT-NEXT:       %c0_2 = arith.constant 0 : index
// RESOPT-NEXT:       %c4 = arith.constant 4 : index
// RESOPT-NEXT:       %c1 = arith.constant 1 : index
// RESOPT-NEXT:       %1 = taskflow.counter from %c0_2 to %c4 step %c1 attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} : index
// RESOPT-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0_2 to %c9 step %c1 attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} : index
// RESOPT-NEXT:       neura.kernel inputs(%arg10, %arg11, %arg12 : memref<?x9xi32>, memref<?xi32>, memref<?xi32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// RESOPT-NEXT:       ^bb0(%arg13: memref<?x9xi32>, %arg14: memref<?xi32>, %arg15: memref<?xi32>):
// RESOPT-NEXT:         %3 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 4 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %4 = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32, lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 9 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %5 = neura.load_indexed [%3, %4 : !neura.data<index, i1>, !neura.data<index, i1>]  {lhs_value = "%input0"} : !neura.data<i32, i1>
// RESOPT-NEXT:         %6 = neura.load_indexed [%4 : !neura.data<index, i1>]  {lhs_value = "%input1"} : !neura.data<i32, i1>
// RESOPT-NEXT:         %7 = "neura.add"(%5, %6) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// RESOPT-NEXT:         neura.store_indexed %7 to [%4 : !neura.data<index, i1>]  {rhs_value = "%input2"} : !neura.data<i32, i1>
// RESOPT-NEXT:         neura.yield {yield_type = "void"}
// RESOPT-NEXT:       }
// RESOPT-NEXT:       taskflow.yield writes(%arg12 : memref<?xi32>)
// RESOPT-NEXT:     }
// RESOPT-NEXT:     %0 = memref.load %dependency_write_out_0#0[%c0] : memref<?xi32>
// RESOPT-NEXT:     return %0 : i32
// RESOPT-NEXT:   }
// RESOPT-NEXT: }