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
// RUN: --allocate-cgra-to-task \
// RUN: -o %t.placement.mlir
// RUN: FileCheck %s --input-file=%t.placement.mlir --check-prefixes=PLACEMENT

// RUN: mlir-neura-opt %s --affine-loop-tree-serialization \
// RUN: --affine-loop-perfection \
// RUN: --convert-affine-to-taskflow \
// RUN: --construct-hyperblock-from-task \
// RUN: --classify-counters \
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

#set = affine_set<(d0, d1) : (d0 - 3 == 0, d1 - 7 == 0)>
module attributes {} {
  func.func @_Z21irregularLoopExample1v() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2_i32 = arith.constant 2 : i32
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<i32>
    %alloca_0 = memref.alloca() : memref<4x8xi32>
    %0 = affine.for %arg0 = 0 to 5 iter_args(%arg1 = %c0_i32) -> (i32) {
      %2 = arith.index_cast %arg0 : index to i32
      %3 = arith.addi %arg1, %2 : i32
      affine.yield %3 : i32
    }
    affine.for %arg0 = 0 to 4 {
      %2 = arith.index_cast %arg0 : index to i32
      %3 = arith.muli %2, %c8_i32 : i32
      affine.for %arg1 = 0 to 8 {
        %4 = arith.index_cast %arg1 : index to i32
        %5 = arith.addi %3, %4 : i32
        affine.store %5, %alloca_0[%arg0, %arg1] : memref<4x8xi32>
      }
      affine.for %arg1 = 0 to 8 {
        %4 = affine.load %alloca_0[%arg0, %arg1] : memref<4x8xi32>
        %5 = arith.addi %4, %0 : i32
        affine.if #set(%arg0, %arg1) {
          affine.store %5, %alloca[] : memref<i32>
          %6 = arith.muli %5, %c2_i32 : i32
          affine.store %6, %alloca[] : memref<i32>
        }
      }
    }
    %1 = affine.load %alloca[] : memref<i32>
    return %1 : i32
  }
}

// SERIALIZED: #set = affine_set<(d0, d1) : (d0 - 3 == 0, d1 - 7 == 0)>
// SERIALIZED-NEXT: module {
// SERIALIZED-NEXT:   func.func @_Z21irregularLoopExample1v() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// SERIALIZED-NEXT:     %c2_i32 = arith.constant 2 : i32
// SERIALIZED-NEXT:     %c8_i32 = arith.constant 8 : i32
// SERIALIZED-NEXT:     %c0_i32 = arith.constant 0 : i32
// SERIALIZED-NEXT:     %alloca = memref.alloca() : memref<i32>
// SERIALIZED-NEXT:     %alloca_0 = memref.alloca() : memref<4x8xi32>
// SERIALIZED-NEXT:     %0 = affine.for %arg0 = 0 to 5 iter_args(%arg1 = %c0_i32) -> (i32) {
// SERIALIZED-NEXT:       %2 = arith.index_cast %arg0 : index to i32
// SERIALIZED-NEXT:       %3 = arith.addi %arg1, %2 : i32
// SERIALIZED-NEXT:       affine.yield %3 : i32
// SERIALIZED-NEXT:     }
// SERIALIZED-NEXT:     affine.for %arg0 = 0 to 4 {
// SERIALIZED-NEXT:       %2 = arith.index_cast %arg0 : index to i32
// SERIALIZED-NEXT:       %3 = arith.muli %2, %c8_i32 : i32
// SERIALIZED-NEXT:       affine.for %arg1 = 0 to 8 {
// SERIALIZED-NEXT:         %4 = arith.index_cast %arg1 : index to i32
// SERIALIZED-NEXT:         %5 = arith.addi %3, %4 : i32
// SERIALIZED-NEXT:         affine.store %5, %alloca_0[%arg0, %arg1] : memref<4x8xi32>
// SERIALIZED-NEXT:       }
// SERIALIZED-NEXT:     }
// SERIALIZED-NEXT:     affine.for %arg0 = 0 to 4 {
// SERIALIZED-NEXT:       %2 = arith.index_cast %arg0 : index to i32
// SERIALIZED-NEXT:       %3 = arith.muli %2, %c8_i32 : i32
// SERIALIZED-NEXT:       affine.for %arg1 = 0 to 8 {
// SERIALIZED-NEXT:         %4 = affine.load %alloca_0[%arg0, %arg1] : memref<4x8xi32>
// SERIALIZED-NEXT:         %5 = arith.addi %4, %0 : i32
// SERIALIZED-NEXT:         affine.if #set(%arg0, %arg1) {
// SERIALIZED-NEXT:           affine.store %5, %alloca[] : memref<i32>
// SERIALIZED-NEXT:           %6 = arith.muli %5, %c2_i32 : i32
// SERIALIZED-NEXT:           affine.store %6, %alloca[] : memref<i32>
// SERIALIZED-NEXT:         }
// SERIALIZED-NEXT:       }
// SERIALIZED-NEXT:     }
// SERIALIZED-NEXT:     %1 = affine.load %alloca[] : memref<i32>
// SERIALIZED-NEXT:     return %1 : i32
// SERIALIZED-NEXT:   }
// SERIALIZED-NEXT: }

// PERFECT: #set = affine_set<(d0, d1) : (d0 - 3 == 0, d1 - 7 == 0)>
// PERFECT-NEXT: module {
// PERFECT-NEXT:   func.func @_Z21irregularLoopExample1v() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// PERFECT-NEXT:     %c2_i32 = arith.constant 2 : i32
// PERFECT-NEXT:     %c8_i32 = arith.constant 8 : i32
// PERFECT-NEXT:     %c0_i32 = arith.constant 0 : i32
// PERFECT-NEXT:     %alloca = memref.alloca() : memref<i32>
// PERFECT-NEXT:     %alloca_0 = memref.alloca() : memref<4x8xi32>
// PERFECT-NEXT:     %0 = affine.for %arg0 = 0 to 5 iter_args(%arg1 = %c0_i32) -> (i32) {
// PERFECT-NEXT:       %2 = arith.index_cast %arg0 : index to i32
// PERFECT-NEXT:       %3 = arith.addi %arg1, %2 : i32
// PERFECT-NEXT:       affine.yield %3 : i32
// PERFECT-NEXT:     }
// PERFECT-NEXT:     affine.for %arg0 = 0 to 4 {
// PERFECT-NEXT:       affine.for %arg1 = 0 to 8 {
// PERFECT-NEXT:         %2 = arith.index_cast %arg0 : index to i32
// PERFECT-NEXT:         %3 = arith.muli %2, %c8_i32 : i32
// PERFECT-NEXT:         %4 = arith.index_cast %arg1 : index to i32
// PERFECT-NEXT:         %5 = arith.addi %3, %4 : i32
// PERFECT-NEXT:         affine.store %5, %alloca_0[%arg0, %arg1] : memref<4x8xi32>
// PERFECT-NEXT:       }
// PERFECT-NEXT:     }
// PERFECT-NEXT:     affine.for %arg0 = 0 to 4 {
// PERFECT-NEXT:       affine.for %arg1 = 0 to 8 {
// PERFECT-NEXT:         %2 = arith.index_cast %arg0 : index to i32
// PERFECT-NEXT:         %3 = arith.muli %2, %c8_i32 : i32
// PERFECT-NEXT:         %4 = affine.load %alloca_0[%arg0, %arg1] : memref<4x8xi32>
// PERFECT-NEXT:         %5 = arith.addi %4, %0 : i32
// PERFECT-NEXT:         affine.if #set(%arg0, %arg1) {
// PERFECT-NEXT:           affine.store %5, %alloca[] : memref<i32>
// PERFECT-NEXT:           %6 = arith.muli %5, %c2_i32 : i32
// PERFECT-NEXT:           affine.store %6, %alloca[] : memref<i32>
// PERFECT-NEXT:         }
// PERFECT-NEXT:       }
// PERFECT-NEXT:     }
// PERFECT-NEXT:     %1 = affine.load %alloca[] : memref<i32>
// PERFECT-NEXT:     return %1 : i32
// PERFECT-NEXT:   }
// PERFECT-NEXT: }

// TASKFLOW: #set = affine_set<(d0, d1) : (d0 - 3 == 0, d1 - 7 == 0)>
// TASKFLOW-NEXT: module {
// TASKFLOW-NEXT:   func.func @_Z21irregularLoopExample1v() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// TASKFLOW-NEXT:     %c2_i32 = arith.constant 2 : i32
// TASKFLOW-NEXT:     %c8_i32 = arith.constant 8 : i32
// TASKFLOW-NEXT:     %c0_i32 = arith.constant 0 : i32
// TASKFLOW-NEXT:     %alloca = memref.alloca() : memref<i32>
// TASKFLOW-NEXT:     %alloca_0 = memref.alloca() : memref<4x8xi32>
// TASKFLOW-NEXT:     %value_outputs = taskflow.task @Task_0 value_inputs(%c0_i32 : i32) : (i32) -> (i32) {
// TASKFLOW-NEXT:     ^bb0(%arg0: i32):
// TASKFLOW-NEXT:       %1 = affine.for %arg1 = 0 to 5 iter_args(%arg2 = %arg0) -> (i32) {
// TASKFLOW-NEXT:         %2 = arith.index_cast %arg1 : index to i32
// TASKFLOW-NEXT:         %3 = arith.addi %arg2, %2 : i32
// TASKFLOW-NEXT:         affine.yield %3 : i32
// TASKFLOW-NEXT:       }
// TASKFLOW-NEXT:       taskflow.yield values(%1 : i32)
// TASKFLOW-NEXT:     }
// TASKFLOW-NEXT:     %dependency_write_out = taskflow.task @Task_1 dependency_write_in(%alloca_0 : memref<4x8xi32>) value_inputs(%c8_i32 : i32) [original_write_memrefs(%alloca_0 : memref<4x8xi32>)] : (memref<4x8xi32>, i32) -> (memref<4x8xi32>) {
// TASKFLOW-NEXT:     ^bb0(%arg0: memref<4x8xi32>, %arg1: i32):
// TASKFLOW-NEXT:       affine.for %arg2 = 0 to 4 {
// TASKFLOW-NEXT:         %1 = arith.index_cast %arg2 : index to i32
// TASKFLOW-NEXT:         %2 = arith.muli %1, %arg1 : i32
// TASKFLOW-NEXT:         affine.for %arg3 = 0 to 8 {
// TASKFLOW-NEXT:           %3 = arith.index_cast %arg3 : index to i32
// TASKFLOW-NEXT:           %4 = arith.addi %2, %3 : i32
// TASKFLOW-NEXT:           affine.store %4, %arg0[%arg2, %arg3] : memref<4x8xi32>
// TASKFLOW-NEXT:         }
// TASKFLOW-NEXT:       }
// TASKFLOW-NEXT:       taskflow.yield writes(%arg0 : memref<4x8xi32>)
// TASKFLOW-NEXT:     }
// TASKFLOW-NEXT:     %dependency_read_out, %dependency_write_out_1 = taskflow.task @Task_2 dependency_read_in(%dependency_write_out : memref<4x8xi32>) dependency_write_in(%alloca : memref<i32>) value_inputs(%c8_i32, %value_outputs, %c2_i32 : i32, i32, i32) [original_read_memrefs(%alloca_0 : memref<4x8xi32>), original_write_memrefs(%alloca : memref<i32>)] : (memref<4x8xi32>, memref<i32>, i32, i32, i32) -> (memref<4x8xi32>, memref<i32>) {
// TASKFLOW-NEXT:     ^bb0(%arg0: memref<4x8xi32>, %arg1: memref<i32>, %arg2: i32, %arg3: i32, %arg4: i32):
// TASKFLOW-NEXT:       affine.for %arg5 = 0 to 4 {
// TASKFLOW-NEXT:         %1 = arith.index_cast %arg5 : index to i32
// TASKFLOW-NEXT:         %2 = arith.muli %1, %arg2 : i32
// TASKFLOW-NEXT:         affine.for %arg6 = 0 to 8 {
// TASKFLOW-NEXT:           %3 = affine.load %arg0[%arg5, %arg6] : memref<4x8xi32>
// TASKFLOW-NEXT:           %4 = arith.addi %3, %arg3 : i32
// TASKFLOW-NEXT:           affine.if #set(%arg5, %arg6) {
// TASKFLOW-NEXT:             affine.store %4, %arg1[] : memref<i32>
// TASKFLOW-NEXT:             %5 = arith.muli %4, %arg4 : i32
// TASKFLOW-NEXT:             affine.store %5, %arg1[] : memref<i32>
// TASKFLOW-NEXT:           }
// TASKFLOW-NEXT:         }
// TASKFLOW-NEXT:       }
// TASKFLOW-NEXT:       taskflow.yield reads(%arg0 : memref<4x8xi32>) writes(%arg1 : memref<i32>)
// TASKFLOW-NEXT:     }
// TASKFLOW-NEXT:     %0 = affine.load %dependency_write_out_1[] : memref<i32>
// TASKFLOW-NEXT:     return %0 : i32
// TASKFLOW-NEXT:   }
// TASKFLOW-NEXT: }

// KERNEL: module {
// KERNEL-NEXT:   func.func @_Z21irregularLoopExample1v() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// KERNEL-NEXT:     %c2_i32 = arith.constant 2 : i32
// KERNEL-NEXT:     %c8_i32 = arith.constant 8 : i32
// KERNEL-NEXT:     %c0_i32 = arith.constant 0 : i32
// KERNEL-NEXT:     %alloca = memref.alloca() : memref<i32>
// KERNEL-NEXT:     %alloca_0 = memref.alloca() : memref<4x8xi32>
// KERNEL-NEXT:     %value_outputs = taskflow.task @Task_0 value_inputs(%c0_i32 : i32) : (i32) -> (i32) {
// KERNEL-NEXT:     ^bb0(%arg0: i32):
// KERNEL-NEXT:       %1 = neura.kernel inputs(%arg0 : i32) {
// KERNEL-NEXT:       ^bb0(%arg1: i32):
// KERNEL-NEXT:         %c0 = arith.constant 0 : index
// KERNEL-NEXT:         %c5 = arith.constant 5 : index
// KERNEL-NEXT:         %c1 = arith.constant 1 : index
// KERNEL-NEXT:         %2 = scf.for %arg2 = %c0 to %c5 step %c1 iter_args(%arg3 = %arg1) -> (i32) {
// KERNEL-NEXT:           %3 = arith.index_cast %arg2 : index to i32
// KERNEL-NEXT:           %4 = arith.addi %arg3, %3 : i32
// KERNEL-NEXT:           scf.yield %4 : i32
// KERNEL-NEXT:         }
// KERNEL-NEXT:         neura.yield results(%2 : i32)
// KERNEL-NEXT:       } : i32
// KERNEL-NEXT:       taskflow.yield values(%1 : i32)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %dependency_write_out = taskflow.task @Task_1 dependency_write_in(%alloca_0 : memref<4x8xi32>) value_inputs(%c8_i32 : i32) [original_write_memrefs(%alloca_0 : memref<4x8xi32>)] : (memref<4x8xi32>, i32) -> (memref<4x8xi32>) {
// KERNEL-NEXT:     ^bb0(%arg0: memref<4x8xi32>, %arg1: i32):
// KERNEL-NEXT:       affine.for %arg2 = 0 to 4 {
// KERNEL-NEXT:         %1 = arith.index_cast %arg2 : index to i32
// KERNEL-NEXT:         %2 = arith.muli %1, %arg1 : i32
// KERNEL-NEXT:         neura.kernel inputs(%2, %arg0, %arg2 : i32, memref<4x8xi32>, index) {
// KERNEL-NEXT:         ^bb0(%arg3: i32, %arg4: memref<4x8xi32>, %arg5: index):
// KERNEL-NEXT:           %c0 = arith.constant 0 : index
// KERNEL-NEXT:           %c8 = arith.constant 8 : index
// KERNEL-NEXT:           %c1 = arith.constant 1 : index
// KERNEL-NEXT:           scf.for %arg6 = %c0 to %c8 step %c1 {
// KERNEL-NEXT:             %3 = arith.index_cast %arg6 : index to i32
// KERNEL-NEXT:             %4 = arith.addi %arg3, %3 : i32
// KERNEL-NEXT:             memref.store %4, %arg4[%arg5, %arg6] : memref<4x8xi32>
// KERNEL-NEXT:           }
// KERNEL-NEXT:           neura.yield
// KERNEL-NEXT:         }
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield writes(%arg0 : memref<4x8xi32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %dependency_read_out, %dependency_write_out_1 = taskflow.task @Task_2 dependency_read_in(%dependency_write_out : memref<4x8xi32>) dependency_write_in(%alloca : memref<i32>) value_inputs(%c8_i32, %value_outputs, %c2_i32 : i32, i32, i32) [original_read_memrefs(%alloca_0 : memref<4x8xi32>), original_write_memrefs(%alloca : memref<i32>)] : (memref<4x8xi32>, memref<i32>, i32, i32, i32) -> (memref<4x8xi32>, memref<i32>) {
// KERNEL-NEXT:     ^bb0(%arg0: memref<4x8xi32>, %arg1: memref<i32>, %arg2: i32, %arg3: i32, %arg4: i32):
// KERNEL-NEXT:       affine.for %arg5 = 0 to 4 {
// KERNEL-NEXT:         neura.kernel inputs(%arg0, %arg5, %arg3, %arg1, %arg4 : memref<4x8xi32>, index, i32, memref<i32>, i32) {
// KERNEL-NEXT:         ^bb0(%arg6: memref<4x8xi32>, %arg7: index, %arg8: i32, %arg9: memref<i32>, %arg10: i32):
// KERNEL-NEXT:           %c-3 = arith.constant -3 : index
// KERNEL-NEXT:           %c-7 = arith.constant -7 : index
// KERNEL-NEXT:           %c0 = arith.constant 0 : index
// KERNEL-NEXT:           %c8 = arith.constant 8 : index
// KERNEL-NEXT:           %c1 = arith.constant 1 : index
// KERNEL-NEXT:           scf.for %arg11 = %c0 to %c8 step %c1 {
// KERNEL-NEXT:             %1 = memref.load %arg6[%arg7, %arg11] : memref<4x8xi32>
// KERNEL-NEXT:             %2 = arith.addi %1, %arg8 : i32
// KERNEL-NEXT:             %3 = arith.addi %arg7, %c-3 : index
// KERNEL-NEXT:             %4 = arith.cmpi eq, %3, %c0 : index
// KERNEL-NEXT:             %5 = arith.addi %arg11, %c-7 : index
// KERNEL-NEXT:             %6 = arith.cmpi eq, %5, %c0 : index
// KERNEL-NEXT:             %7 = arith.andi %4, %6 : i1
// KERNEL-NEXT:             scf.if %7 {
// KERNEL-NEXT:               memref.store %2, %arg9[] : memref<i32>
// KERNEL-NEXT:               %8 = arith.muli %2, %arg10 : i32
// KERNEL-NEXT:               memref.store %8, %arg9[] : memref<i32>
// KERNEL-NEXT:             }
// KERNEL-NEXT:           }
// KERNEL-NEXT:           neura.yield
// KERNEL-NEXT:         }
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield reads(%arg0 : memref<4x8xi32>) writes(%arg1 : memref<i32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %0 = affine.load %dependency_write_out_1[] : memref<i32>
// KERNEL-NEXT:     return %0 : i32
// KERNEL-NEXT:   }
// KERNEL-NEXT: }

// HYPERBLOCK:      module {
// HYPERBLOCK-NEXT:   func.func @_Z21irregularLoopExample1v() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// HYPERBLOCK-NEXT:     %c2_i32 = arith.constant 2 : i32
// HYPERBLOCK-NEXT:     %c8_i32 = arith.constant 8 : i32
// HYPERBLOCK-NEXT:     %c0_i32 = arith.constant 0 : i32
// HYPERBLOCK-NEXT:     %alloca = memref.alloca() : memref<i32>
// HYPERBLOCK-NEXT:     %alloca_0 = memref.alloca() : memref<4x8xi32>
// HYPERBLOCK-NEXT:     %value_outputs = taskflow.task @Task_0 value_inputs(%c0_i32 : i32) : (i32) -> (i32) {
// HYPERBLOCK-NEXT:     ^bb0(%arg0: i32):
// HYPERBLOCK-NEXT:       %c0 = arith.constant 0 : index
// HYPERBLOCK-NEXT:       %c5 = arith.constant 5 : index
// HYPERBLOCK-NEXT:       %c1 = arith.constant 1 : index
// HYPERBLOCK-NEXT:       %1 = taskflow.counter from %c0 to %c5 step %c1 : index
// HYPERBLOCK-NEXT:       %2 = "taskflow.hyperblock"(%1, %arg0) <{operandSegmentSizes = array<i32: 1, 1>}> ({
// HYPERBLOCK-NEXT:       ^bb0(%arg1: index, %arg2: i32):
// HYPERBLOCK-NEXT:         %3 = arith.index_cast %arg1 : index to i32
// HYPERBLOCK-NEXT:         %4 = arith.addi %arg2, %3 : i32
// HYPERBLOCK-NEXT:         taskflow.hyperblock.yield iter_args_next(%4 : i32) results(%4 : i32)
// HYPERBLOCK-NEXT:       }) : (index, i32) -> i32
// HYPERBLOCK-NEXT:       taskflow.yield values(%2 : i32)
// HYPERBLOCK-NEXT:     }
// HYPERBLOCK-NEXT:     %dependency_write_out = taskflow.task @Task_1 dependency_write_in(%alloca_0 : memref<4x8xi32>) value_inputs(%c8_i32 : i32) [original_write_memrefs(%alloca_0 : memref<4x8xi32>)] : (memref<4x8xi32>, i32) -> (memref<4x8xi32>) {
// HYPERBLOCK-NEXT:     ^bb0(%arg0: memref<4x8xi32>, %arg1: i32):
// HYPERBLOCK-NEXT:       %c0 = arith.constant 0 : index
// HYPERBLOCK-NEXT:       %c4 = arith.constant 4 : index
// HYPERBLOCK-NEXT:       %c1 = arith.constant 1 : index
// HYPERBLOCK-NEXT:       %1 = taskflow.counter from %c0 to %c4 step %c1 : index
// HYPERBLOCK-NEXT:       "taskflow.hyperblock"(%1) <{operandSegmentSizes = array<i32: 1, 0>}> ({
// HYPERBLOCK-NEXT:       ^bb0(%arg2: index):
// HYPERBLOCK-NEXT:         %2 = arith.index_cast %arg2 : index to i32
// HYPERBLOCK-NEXT:         %3 = arith.muli %2, %arg1 : i32
// HYPERBLOCK-NEXT:         %c0_2 = arith.constant 0 : index
// HYPERBLOCK-NEXT:         %c8 = arith.constant 8 : index
// HYPERBLOCK-NEXT:         %c1_3 = arith.constant 1 : index
// HYPERBLOCK-NEXT:         scf.for %arg3 = %c0_2 to %c8 step %c1_3 {
// HYPERBLOCK-NEXT:           %4 = arith.index_cast %arg3 : index to i32
// HYPERBLOCK-NEXT:           %5 = arith.addi %3, %4 : i32
// HYPERBLOCK-NEXT:           memref.store %5, %arg0[%arg2, %arg3] : memref<4x8xi32>
// HYPERBLOCK-NEXT:         }
// HYPERBLOCK-NEXT:         taskflow.hyperblock.yield
// HYPERBLOCK-NEXT:       }) : (index) -> ()
// HYPERBLOCK-NEXT:       taskflow.yield writes(%arg0 : memref<4x8xi32>)
// HYPERBLOCK-NEXT:     }
// HYPERBLOCK-NEXT:     %dependency_read_out, %dependency_write_out_1 = taskflow.task @Task_2 dependency_read_in(%dependency_write_out : memref<4x8xi32>) dependency_write_in(%alloca : memref<i32>) value_inputs(%c8_i32, %value_outputs, %c2_i32 : i32, i32, i32) [original_read_memrefs(%alloca_0 : memref<4x8xi32>), original_write_memrefs(%alloca : memref<i32>)] : (memref<4x8xi32>, memref<i32>, i32, i32, i32) -> (memref<4x8xi32>, memref<i32>) {
// HYPERBLOCK-NEXT:     ^bb0(%arg0: memref<4x8xi32>, %arg1: memref<i32>, %arg2: i32, %arg3: i32, %arg4: i32):
// HYPERBLOCK-NEXT:       %c0 = arith.constant 0 : index
// HYPERBLOCK-NEXT:       %c4 = arith.constant 4 : index
// HYPERBLOCK-NEXT:       %c1 = arith.constant 1 : index
// HYPERBLOCK-NEXT:       %1 = taskflow.counter from %c0 to %c4 step %c1 : index
// HYPERBLOCK-NEXT:       "taskflow.hyperblock"(%1) <{operandSegmentSizes = array<i32: 1, 0>}> ({
// HYPERBLOCK-NEXT:       ^bb0(%arg5: index):
// HYPERBLOCK-NEXT:         %2 = arith.index_cast %arg5 : index to i32
// HYPERBLOCK-NEXT:         %3 = arith.muli %2, %arg2 : i32
// HYPERBLOCK-NEXT:         %c0_2 = arith.constant 0 : index
// HYPERBLOCK-NEXT:         %c8 = arith.constant 8 : index
// HYPERBLOCK-NEXT:         %c1_3 = arith.constant 1 : index
// HYPERBLOCK-NEXT:         scf.for %arg6 = %c0_2 to %c8 step %c1_3 {
// HYPERBLOCK-NEXT:           %4 = memref.load %arg0[%arg5, %arg6] : memref<4x8xi32>
// HYPERBLOCK-NEXT:           %5 = arith.addi %4, %arg3 : i32
// HYPERBLOCK-NEXT:           %c0_4 = arith.constant 0 : index
// HYPERBLOCK-NEXT:           %c-3 = arith.constant -3 : index
// HYPERBLOCK-NEXT:           %6 = arith.addi %arg5, %c-3 : index
// HYPERBLOCK-NEXT:           %7 = arith.cmpi eq, %6, %c0_4 : index
// HYPERBLOCK-NEXT:           %c-7 = arith.constant -7 : index
// HYPERBLOCK-NEXT:           %8 = arith.addi %arg6, %c-7 : index
// HYPERBLOCK-NEXT:           %9 = arith.cmpi eq, %8, %c0_4 : index
// HYPERBLOCK-NEXT:           %10 = arith.andi %7, %9 : i1
// HYPERBLOCK-NEXT:           scf.if %10 {
// HYPERBLOCK-NEXT:             memref.store %5, %arg1[] : memref<i32>
// HYPERBLOCK-NEXT:             %11 = arith.muli %5, %arg4 : i32
// HYPERBLOCK-NEXT:             memref.store %11, %arg1[] : memref<i32>
// HYPERBLOCK-NEXT:           }
// HYPERBLOCK-NEXT:         }
// HYPERBLOCK-NEXT:         taskflow.hyperblock.yield
// HYPERBLOCK-NEXT:       }) : (index) -> ()
// HYPERBLOCK-NEXT:       taskflow.yield reads(%arg0 : memref<4x8xi32>) writes(%arg1 : memref<i32>)
// HYPERBLOCK-NEXT:     }
// HYPERBLOCK-NEXT:     %0 = affine.load %dependency_write_out_1[] : memref<i32>
// HYPERBLOCK-NEXT:     return %0 : i32
// HYPERBLOCK-NEXT:   }
// HYPERBLOCK-NEXT: }

// PLACEMENT:      module {
// PLACEMENT-NEXT:   func.func @_Z21irregularLoopExample1v() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// PLACEMENT-NEXT:     %c2_i32 = arith.constant 2 : i32
// PLACEMENT-NEXT:     %c8_i32 = arith.constant 8 : i32
// PLACEMENT-NEXT:     %c0_i32 = arith.constant 0 : i32
// PLACEMENT-NEXT:     %alloca = memref.alloca() : memref<i32>
// PLACEMENT-NEXT:     %alloca_0 = memref.alloca() : memref<4x8xi32>
// PLACEMENT-NEXT:     %value_outputs = taskflow.task @Task_0 value_inputs(%c0_i32 : i32) {task_allocation_info = {cgra_positions = [{col = 0 : i32, row = 0 : i32}], read_sram_locations = [], write_sram_locations = []}} : (i32) -> (i32) {
// PLACEMENT-NEXT:     ^bb0(%arg0: i32):
// PLACEMENT-NEXT:       %c0 = arith.constant 0 : index
// PLACEMENT-NEXT:       %c5 = arith.constant 5 : index
// PLACEMENT-NEXT:       %c1 = arith.constant 1 : index
// PLACEMENT-NEXT:       %1 = taskflow.counter from %c0 to %c5 step %c1 : index
// PLACEMENT-NEXT:       %2 = "taskflow.hyperblock"(%1, %arg0) <{operandSegmentSizes = array<i32: 1, 1>}> ({
// PLACEMENT-NEXT:       ^bb0(%arg1: index, %arg2: i32):
// PLACEMENT-NEXT:         %3 = arith.index_cast %arg1 : index to i32
// PLACEMENT-NEXT:         %4 = arith.addi %arg2, %3 : i32
// PLACEMENT-NEXT:         taskflow.hyperblock.yield iter_args_next(%4 : i32) results(%4 : i32)
// PLACEMENT-NEXT:       }) : (index, i32) -> i32
// PLACEMENT-NEXT:       taskflow.yield values(%2 : i32)
// PLACEMENT-NEXT:     }
// PLACEMENT-NEXT:     %dependency_write_out = taskflow.task @Task_1 dependency_write_in(%alloca_0 : memref<4x8xi32>) value_inputs(%c8_i32 : i32) [original_write_memrefs(%alloca_0 : memref<4x8xi32>)] {task_allocation_info = {cgra_positions = [{col = 2 : i32, row = 0 : i32}], read_sram_locations = [], write_sram_locations = [{col = 2 : i32, row = 0 : i32}]}} : (memref<4x8xi32>, i32) -> (memref<4x8xi32>) {
// PLACEMENT-NEXT:     ^bb0(%arg0: memref<4x8xi32>, %arg1: i32):
// PLACEMENT-NEXT:       %c0 = arith.constant 0 : index
// PLACEMENT-NEXT:       %c4 = arith.constant 4 : index
// PLACEMENT-NEXT:       %c1 = arith.constant 1 : index
// PLACEMENT-NEXT:       %1 = taskflow.counter from %c0 to %c4 step %c1 : index
// PLACEMENT-NEXT:       "taskflow.hyperblock"(%1) <{operandSegmentSizes = array<i32: 1, 0>}> ({
// PLACEMENT-NEXT:       ^bb0(%arg2: index):
// PLACEMENT-NEXT:         %2 = arith.index_cast %arg2 : index to i32
// PLACEMENT-NEXT:         %3 = arith.muli %2, %arg1 : i32
// PLACEMENT-NEXT:         %c0_2 = arith.constant 0 : index
// PLACEMENT-NEXT:         %c8 = arith.constant 8 : index
// PLACEMENT-NEXT:         %c1_3 = arith.constant 1 : index
// PLACEMENT-NEXT:         scf.for %arg3 = %c0_2 to %c8 step %c1_3 {
// PLACEMENT-NEXT:           %4 = arith.index_cast %arg3 : index to i32
// PLACEMENT-NEXT:           %5 = arith.addi %3, %4 : i32
// PLACEMENT-NEXT:           memref.store %5, %arg0[%arg2, %arg3] : memref<4x8xi32>
// PLACEMENT-NEXT:         }
// PLACEMENT-NEXT:         taskflow.hyperblock.yield
// PLACEMENT-NEXT:       }) : (index) -> ()
// PLACEMENT-NEXT:       taskflow.yield writes(%arg0 : memref<4x8xi32>)
// PLACEMENT-NEXT:     }
// PLACEMENT-NEXT:     %dependency_read_out, %dependency_write_out_1 = taskflow.task @Task_2 dependency_read_in(%dependency_write_out : memref<4x8xi32>) dependency_write_in(%alloca : memref<i32>) value_inputs(%c8_i32, %value_outputs, %c2_i32 : i32, i32, i32) [original_read_memrefs(%alloca_0 : memref<4x8xi32>), original_write_memrefs(%alloca : memref<i32>)] {task_allocation_info = {cgra_positions = [{col = 1 : i32, row = 0 : i32}], read_sram_locations = [{col = 2 : i32, row = 0 : i32}], write_sram_locations = [{col = 1 : i32, row = 0 : i32}]}} : (memref<4x8xi32>, memref<i32>, i32, i32, i32) -> (memref<4x8xi32>, memref<i32>) {
// PLACEMENT-NEXT:     ^bb0(%arg0: memref<4x8xi32>, %arg1: memref<i32>, %arg2: i32, %arg3: i32, %arg4: i32):
// PLACEMENT-NEXT:       %c0 = arith.constant 0 : index
// PLACEMENT-NEXT:       %c4 = arith.constant 4 : index
// PLACEMENT-NEXT:       %c1 = arith.constant 1 : index
// PLACEMENT-NEXT:       %1 = taskflow.counter from %c0 to %c4 step %c1 : index
// PLACEMENT-NEXT:       "taskflow.hyperblock"(%1) <{operandSegmentSizes = array<i32: 1, 0>}> ({
// PLACEMENT-NEXT:       ^bb0(%arg5: index):
// PLACEMENT-NEXT:         %2 = arith.index_cast %arg5 : index to i32
// PLACEMENT-NEXT:         %3 = arith.muli %2, %arg2 : i32
// PLACEMENT-NEXT:         %c0_2 = arith.constant 0 : index
// PLACEMENT-NEXT:         %c8 = arith.constant 8 : index
// PLACEMENT-NEXT:         %c1_3 = arith.constant 1 : index
// PLACEMENT-NEXT:         scf.for %arg6 = %c0_2 to %c8 step %c1_3 {
// PLACEMENT-NEXT:           %4 = memref.load %arg0[%arg5, %arg6] : memref<4x8xi32>
// PLACEMENT-NEXT:           %5 = arith.addi %4, %arg3 : i32
// PLACEMENT-NEXT:           %c0_4 = arith.constant 0 : index
// PLACEMENT-NEXT:           %c-3 = arith.constant -3 : index
// PLACEMENT-NEXT:           %6 = arith.addi %arg5, %c-3 : index
// PLACEMENT-NEXT:           %7 = arith.cmpi eq, %6, %c0_4 : index
// PLACEMENT-NEXT:           %c-7 = arith.constant -7 : index
// PLACEMENT-NEXT:           %8 = arith.addi %arg6, %c-7 : index
// PLACEMENT-NEXT:           %9 = arith.cmpi eq, %8, %c0_4 : index
// PLACEMENT-NEXT:           %10 = arith.andi %7, %9 : i1
// PLACEMENT-NEXT:           scf.if %10 {
// PLACEMENT-NEXT:             memref.store %5, %arg1[] : memref<i32>
// PLACEMENT-NEXT:             %11 = arith.muli %5, %arg4 : i32
// PLACEMENT-NEXT:             memref.store %11, %arg1[] : memref<i32>
// PLACEMENT-NEXT:           }
// PLACEMENT-NEXT:         }
// PLACEMENT-NEXT:         taskflow.hyperblock.yield
// PLACEMENT-NEXT:       }) : (index) -> ()
// PLACEMENT-NEXT:       taskflow.yield reads(%arg0 : memref<4x8xi32>) writes(%arg1 : memref<i32>)
// PLACEMENT-NEXT:     }
// PLACEMENT-NEXT:     %0 = affine.load %dependency_write_out_1[] : memref<i32>
// PLACEMENT-NEXT:     return %0 : i32
// PLACEMENT-NEXT:   }
// PLACEMENT-NEXT: }

// RESOPT:      module {
// RESOPT-NEXT:   func.func @_Z21irregularLoopExample1v() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// RESOPT-NEXT:     %c2_i32 = arith.constant 2 : i32
// RESOPT-NEXT:     %c8_i32 = arith.constant 8 : i32
// RESOPT-NEXT:     %c0_i32 = arith.constant 0 : i32
// RESOPT-NEXT:     %alloca = memref.alloca() : memref<i32>
// RESOPT-NEXT:     %alloca_0 = memref.alloca() : memref<4x8xi32>
// RESOPT-NEXT:     %dependency_write_out, %value_outputs = taskflow.task @Task_0_Task_1_utilfused dependency_write_in(%alloca_0 : memref<4x8xi32>) value_inputs(%c0_i32, %c8_i32 : i32, i32) [original_write_memrefs(%alloca_0 : memref<4x8xi32>)] {cgra_count = 2 : i32, cgra_shape = "1x2", compiled_ii = 3 : i32, steps = 5 : i32, trip_count = 32 : i32} : (memref<4x8xi32>, i32, i32) -> (memref<4x8xi32>, i32) {
// RESOPT-NEXT:     ^bb0(%arg0: memref<4x8xi32>, %arg1: i32, %arg2: i32):
// RESOPT-NEXT:       %c0 = arith.constant 0 : index
// RESOPT-NEXT:       %c5 = arith.constant 5 : index
// RESOPT-NEXT:       %c1 = arith.constant 1 : index
// RESOPT-NEXT:       %1 = taskflow.counter from %c0 to %c5 step %c1 attributes {counter_id = 0 : i32, counter_type = "leaf"} : index
// RESOPT-NEXT:       %2 = neura.kernel inputs(%arg2, %arg0 : i32, memref<4x8xi32>) iter_args_init(%arg1 : i32) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// RESOPT-NEXT:       ^bb0(%arg3: i32, %arg4: memref<4x8xi32>, %arg5: i32):
// RESOPT-NEXT:         %5 = "neura.grant_once"() <{constant_value = "%iter_arg_init0"}> : () -> !neura.data<i32, i1>
// RESOPT-NEXT:         %6 = neura.reserve : !neura.data<i32, i1>
// RESOPT-NEXT:         %7 = neura.phi_start %5, %6 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// RESOPT-NEXT:         %8 = neura.counter attributes {counter_id = 0 : i32, counter_type = "leaf", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 5 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %9 = "neura.cast"(%8) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i32, i1>
// RESOPT-NEXT:         %10 = "neura.add"(%7, %9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// RESOPT-NEXT:         neura.ctrl_mov %10 -> %6 : !neura.data<i32, i1> !neura.data<i32, i1>
// RESOPT-NEXT:         %11 = neura.extract_predicate %8 : !neura.data<index, i1> -> !neura.data<i1, i1>
// RESOPT-NEXT:         %12 = "neura.not"(%11) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// RESOPT-NEXT:         %13 = neura.grant_predicate %7, %12 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// RESOPT-NEXT:         neura.return_value %13 : !neura.data<i32, i1>
// RESOPT-NEXT:         %14 = neura.counter attributes {counter_id = 0 : i32, counter_type = "root", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 4 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %15 = neura.counter attributes {counter_id = 1 : i32, counter_type = "leaf", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 8 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %16 = "neura.cast"(%14) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i32, i1>
// RESOPT-NEXT:         %17 = "neura.mul"(%16) {rhs_value = "%input0"} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// RESOPT-NEXT:         %18 = "neura.cast"(%15) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i32, i1>
// RESOPT-NEXT:         %19 = "neura.add"(%17, %18) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// RESOPT-NEXT:         neura.store_indexed %19 to [%14, %15 : !neura.data<index, i1>, !neura.data<index, i1>]  {rhs_value = "%input1"} : !neura.data<i32, i1>
// RESOPT-NEXT:         neura.yield
// RESOPT-NEXT:       } : i32
// RESOPT-NEXT:       %c8 = arith.constant 8 : index
// RESOPT-NEXT:       %c0_2 = arith.constant 0 : index
// RESOPT-NEXT:       %c4 = arith.constant 4 : index
// RESOPT-NEXT:       %c1_3 = arith.constant 1 : index
// RESOPT-NEXT:       %3 = taskflow.counter from %c0_2 to %c4 step %c1_3 attributes {counter_id = 0 : i32, counter_type = "root"} : index
// RESOPT-NEXT:       %4 = taskflow.counter parent(%3 : index) from %c0_2 to %c8 step %c1_3 attributes {counter_id = 1 : i32, counter_type = "leaf"} : index
// RESOPT-NEXT:       taskflow.yield writes(%arg0 : memref<4x8xi32>) values(%2 : i32)
// RESOPT-NEXT:     }
// RESOPT-NEXT:     %dependency_read_out, %dependency_write_out_1 = taskflow.task @Task_2 dependency_read_in(%dependency_write_out : memref<4x8xi32>) dependency_write_in(%alloca : memref<i32>) value_inputs(%c8_i32, %value_outputs, %c2_i32 : i32, i32, i32) [original_read_memrefs(%alloca_0 : memref<4x8xi32>), original_write_memrefs(%alloca : memref<i32>)] {cgra_count = 2 : i32, cgra_shape = "1x2", compiled_ii = 2 : i32, steps = 7 : i32, trip_count = 32 : i32} : (memref<4x8xi32>, memref<i32>, i32, i32, i32) -> (memref<4x8xi32>, memref<i32>) {
// RESOPT-NEXT:     ^bb0(%arg0: memref<4x8xi32>, %arg1: memref<i32>, %arg2: i32, %arg3: i32, %arg4: i32):
// RESOPT-NEXT:       %c8 = arith.constant 8 : index
// RESOPT-NEXT:       %c0 = arith.constant 0 : index
// RESOPT-NEXT:       %c4 = arith.constant 4 : index
// RESOPT-NEXT:       %c1 = arith.constant 1 : index
// RESOPT-NEXT:       %1 = taskflow.counter from %c0 to %c4 step %c1 attributes {counter_id = 0 : i32, counter_type = "root"} : index
// RESOPT-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0 to %c8 step %c1 attributes {counter_id = 1 : i32, counter_type = "leaf"} : index
// RESOPT-NEXT:       neura.kernel inputs(%arg0, %arg3, %arg1, %arg4 : memref<4x8xi32>, i32, memref<i32>, i32) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// RESOPT-NEXT:       ^bb0(%arg5: memref<4x8xi32>, %arg6: i32, %arg7: memref<i32>, %arg8: i32):
// RESOPT-NEXT:         %3 = neura.counter attributes {counter_id = 0 : i32, counter_type = "root", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 4 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %4 = neura.counter attributes {counter_id = 1 : i32, counter_type = "leaf", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 8 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %5 = neura.load_indexed [%3, %4 : !neura.data<index, i1>, !neura.data<index, i1>]  {lhs_value = "%input0"} : !neura.data<i32, i1>
// RESOPT-NEXT:         %6 = "neura.add"(%5) {rhs_value = "%input1"} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// RESOPT-NEXT:         %7 = "neura.add"(%3) {rhs_value = -3 : index} : (!neura.data<index, i1>) -> !neura.data<index, i1>
// RESOPT-NEXT:         %8 = "neura.icmp"(%7) <{cmpType = "eq"}> {rhs_value = 0 : index} : (!neura.data<index, i1>) -> !neura.data<i1, i1>
// RESOPT-NEXT:         %9 = "neura.add"(%4) {rhs_value = -7 : index} : (!neura.data<index, i1>) -> !neura.data<index, i1>
// RESOPT-NEXT:         %10 = "neura.icmp"(%9) <{cmpType = "eq"}> {rhs_value = 0 : index} : (!neura.data<index, i1>) -> !neura.data<i1, i1>
// RESOPT-NEXT:         %11 = "neura.and"(%8, %10) : (!neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<i1, i1>
// RESOPT-NEXT:         %12 = neura.grant_predicate %6, %11 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// RESOPT-NEXT:         neura.store_indexed %12 to [ : ]  {rhs_value = "%input2"} : !neura.data<i32, i1>
// RESOPT-NEXT:         %13 = "neura.mul"(%12) {rhs_value = "%input3"} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// RESOPT-NEXT:         neura.store_indexed %13 to [ : ]  {rhs_value = "%input2"} : !neura.data<i32, i1>
// RESOPT-NEXT:         neura.yield {yield_type = "void"}
// RESOPT-NEXT:       }
// RESOPT-NEXT:       taskflow.yield reads(%arg0 : memref<4x8xi32>) writes(%arg1 : memref<i32>)
// RESOPT-NEXT:     }
// RESOPT-NEXT:     %0 = memref.load %dependency_write_out_1[] : memref<i32>
// RESOPT-NEXT:     return %0 : i32
// RESOPT-NEXT:   }
// RESOPT-NEXT: }



