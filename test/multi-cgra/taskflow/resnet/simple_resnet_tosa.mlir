// RUN: neura-compiler %s --tosa-to-affine-conversion \
// RUN: -o %t.affine.mlir
// RUN: FileCheck %s --input-file=%t.affine.mlir --check-prefixes=AFFINE

// RUN: neura-compiler %s --tosa-to-affine-conversion \
// RUN: --taskflow-conversion \
// RUN: -o %t.kernel.mlir
// RUN: FileCheck %s --input-file=%t.kernel.mlir --check-prefixes=KERNEL

// RUN: mlir-neura-opt %t.affine.mlir \
// RUN: --affine-loop-tree-serialization \
// RUN: --affine-loop-perfection \
// RUN: --convert-affine-to-taskflow \
// RUN: --memory-access-streaming-fusion \
// RUN: -o %t.stream.mlir
// RUN: FileCheck %s --input-file=%t.stream.mlir --check-prefixes=STREAM

// RUN: mlir-neura-opt %t.stream.mlir \
// RUN: --affine-loop-tree-serialization \
// RUN: --affine-loop-perfection \
// RUN: --construct-hyperblock-from-task \
// RUN: --classify-counters \
// RUN: --convert-taskflow-to-neura \
// RUN: --cse \
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


module attributes {torch.debug_module_name = "SimpleResNetBlock"} {
  func.func @forward(%arg0: tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32> {
    %0 = "tosa.const"() <{value = dense<"0x7BEEA13C"> : tensor<64x64x3x3xf32>}> : () -> tensor<64x64x3x3xf32>
    %1 = "tosa.const"() <{value = dense<"0x8B9878BC"> : tensor<64x64x3x3xf32>}> : () -> tensor<64x64x3x3xf32>
    %2 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %3 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5 = tosa.transpose %arg0, %3 : (tensor<1x64x8x8xf32>, tensor<4xi32>) -> tensor<1x8x8x64xf32>
    %6 = tosa.transpose %1, %3 : (tensor<64x64x3x3xf32>, tensor<4xi32>) -> tensor<64x3x3x64xf32>
    %7 = tosa.conv2d %5, %6, %2 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x8x8x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<1x8x8x64xf32>
    %8 = tosa.transpose %7, %4 : (tensor<1x8x8x64xf32>, tensor<4xi32>) -> tensor<1x64x8x8xf32>
    %9 = tosa.clamp %8 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
    %10 = tosa.transpose %9, %3 : (tensor<1x64x8x8xf32>, tensor<4xi32>) -> tensor<1x8x8x64xf32>
    %11 = tosa.transpose %0, %3 : (tensor<64x64x3x3xf32>, tensor<4xi32>) -> tensor<64x3x3x64xf32>
    %12 = tosa.conv2d %10, %11, %2 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x8x8x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<1x8x8x64xf32>
    %13 = tosa.transpose %12, %4 : (tensor<1x8x8x64xf32>, tensor<4xi32>) -> tensor<1x64x8x8xf32>
    %14 = tosa.add %13, %arg0 : (tensor<1x64x8x8xf32>, tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
    %15 = tosa.clamp %14 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
    return %15 : tensor<1x64x8x8xf32>
  }
}

// AFFINE: module attributes {torch.debug_module_name = "SimpleResNetBlock"} {
// AFFINE-NEXT:   memref.global "private" constant @__constant_64xf32 : memref<64xf32> = dense<0.000000e+00> {alignment = 64 : i64}
// AFFINE-NEXT:   memref.global "private" constant @__constant_64x3x3x64xf32_0 : memref<64x3x3x64xf32> = dense<-0.0151730878> {alignment = 64 : i64}
// AFFINE-NEXT:   memref.global "private" constant @__constant_64x3x3x64xf32 : memref<64x3x3x64xf32> = dense<0.0197670367> {alignment = 64 : i64}
// AFFINE-NEXT:   func.func @forward(%arg0: memref<1x64x8x8xf32>) -> memref<1x64x8x8xf32> {
// AFFINE-NEXT:     %cst = arith.constant 0.0197670367 : f32
// AFFINE-NEXT:     %cst_0 = arith.constant -0.0151730878 : f32
// AFFINE-NEXT:     %cst_1 = arith.constant 3.40282347E+38 : f32
// AFFINE-NEXT:     %cst_2 = arith.constant 0.000000e+00 : f32
// AFFINE-NEXT:     %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x8x8x64xf32>
// AFFINE-NEXT:     affine.for %arg1 = 0 to 1 {
// AFFINE-NEXT:       affine.for %arg2 = 0 to 8 {
// AFFINE-NEXT:         affine.for %arg3 = 0 to 8 {
// AFFINE-NEXT:           affine.for %arg4 = 0 to 64 {
// AFFINE-NEXT:             %0 = affine.load %arg0[%arg1, %arg4, %arg2, %arg3] : memref<1x64x8x8xf32>
// AFFINE-NEXT:             affine.store %0, %alloc[%arg1, %arg2, %arg3, %arg4] : memref<1x8x8x64xf32>
// AFFINE-NEXT:           }
// AFFINE-NEXT:         }
// AFFINE-NEXT:       }
// AFFINE-NEXT:     }
// AFFINE-NEXT:     %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x10x10x64xf32>
// AFFINE-NEXT:     affine.for %arg1 = 0 to 1 {
// AFFINE-NEXT:       affine.for %arg2 = 0 to 10 {
// AFFINE-NEXT:         affine.for %arg3 = 0 to 10 {
// AFFINE-NEXT:           affine.for %arg4 = 0 to 64 {
// AFFINE-NEXT:             affine.store %cst_2, %alloc_3[%arg1, %arg2, %arg3, %arg4] : memref<1x10x10x64xf32>
// AFFINE-NEXT:           }
// AFFINE-NEXT:         }
// AFFINE-NEXT:       }
// AFFINE-NEXT:     }
// AFFINE-NEXT:     %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1x8x8x64xf32>
// AFFINE-NEXT:     affine.for %arg1 = 0 to 1 {
// AFFINE-NEXT:       affine.for %arg2 = 0 to 8 {
// AFFINE-NEXT:         affine.for %arg3 = 0 to 8 {
// AFFINE-NEXT:           affine.for %arg4 = 0 to 64 {
// AFFINE-NEXT:             affine.store %cst_2, %alloc_4[%arg1, %arg2, %arg3, %arg4] : memref<1x8x8x64xf32>
// AFFINE-NEXT:           }
// AFFINE-NEXT:         }
// AFFINE-NEXT:       }
// AFFINE-NEXT:     }
// AFFINE-NEXT:     affine.for %arg1 = 0 to 1 {
// AFFINE-NEXT:       affine.for %arg2 = 0 to 8 {
// AFFINE-NEXT:         affine.for %arg3 = 0 to 8 {
// AFFINE-NEXT:           affine.for %arg4 = 0 to 64 {
// AFFINE-NEXT:             affine.for %arg5 = 0 to 3 {
// AFFINE-NEXT:               affine.for %arg6 = 0 to 3 {
// AFFINE-NEXT:                 affine.for %arg7 = 0 to 64 {
// AFFINE-NEXT:                   %0 = affine.load %alloc_3[%arg1, %arg2 + %arg5, %arg3 + %arg6, %arg7] : memref<1x10x10x64xf32>
// AFFINE-NEXT:                   %1 = affine.load %alloc_4[%arg1, %arg2, %arg3, %arg4] : memref<1x8x8x64xf32>
// AFFINE-NEXT:                   %2 = arith.mulf %0, %cst_0 : f32
// AFFINE-NEXT:                   %3 = arith.addf %1, %2 : f32
// AFFINE-NEXT:                   affine.store %3, %alloc_4[%arg1, %arg2, %arg3, %arg4] : memref<1x8x8x64xf32>
// AFFINE-NEXT:                 }
// AFFINE-NEXT:               }
// AFFINE-NEXT:             }
// AFFINE-NEXT:           }
// AFFINE-NEXT:         }
// AFFINE-NEXT:       }
// AFFINE-NEXT:     }
// AFFINE-NEXT:     %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x64x8x8xf32>
// AFFINE-NEXT:     affine.for %arg1 = 0 to 1 {
// AFFINE-NEXT:       affine.for %arg2 = 0 to 64 {
// AFFINE-NEXT:         affine.for %arg3 = 0 to 8 {
// AFFINE-NEXT:           affine.for %arg4 = 0 to 8 {
// AFFINE-NEXT:             %0 = affine.load %alloc_4[%arg1, %arg3, %arg4, %arg2] : memref<1x8x8x64xf32>
// AFFINE-NEXT:             affine.store %0, %alloc_5[%arg1, %arg2, %arg3, %arg4] : memref<1x64x8x8xf32>
// AFFINE-NEXT:           }
// AFFINE-NEXT:         }
// AFFINE-NEXT:       }
// AFFINE-NEXT:     }
// AFFINE-NEXT:     %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x64x8x8xf32>
// AFFINE-NEXT:     affine.for %arg1 = 0 to 1 {
// AFFINE-NEXT:       affine.for %arg2 = 0 to 64 {
// AFFINE-NEXT:         affine.for %arg3 = 0 to 8 {
// AFFINE-NEXT:           affine.for %arg4 = 0 to 8 {
// AFFINE-NEXT:             %0 = affine.load %alloc_5[%arg1, %arg2, %arg3, %arg4] : memref<1x64x8x8xf32>
// AFFINE-NEXT:             %1 = arith.minimumf %0, %cst_1 : f32
// AFFINE-NEXT:             %2 = arith.maximumf %1, %cst_2 : f32
// AFFINE-NEXT:             affine.store %2, %alloc_6[%arg1, %arg2, %arg3, %arg4] : memref<1x64x8x8xf32>
// AFFINE-NEXT:           }
// AFFINE-NEXT:         }
// AFFINE-NEXT:       }
// AFFINE-NEXT:     }
// AFFINE-NEXT:     %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x8x8x64xf32>
// AFFINE-NEXT:     affine.for %arg1 = 0 to 1 {
// AFFINE-NEXT:       affine.for %arg2 = 0 to 8 {
// AFFINE-NEXT:         affine.for %arg3 = 0 to 8 {
// AFFINE-NEXT:           affine.for %arg4 = 0 to 64 {
// AFFINE-NEXT:             %0 = affine.load %alloc_6[%arg1, %arg4, %arg2, %arg3] : memref<1x64x8x8xf32>
// AFFINE-NEXT:             affine.store %0, %alloc_7[%arg1, %arg2, %arg3, %arg4] : memref<1x8x8x64xf32>
// AFFINE-NEXT:           }
// AFFINE-NEXT:         }
// AFFINE-NEXT:       }
// AFFINE-NEXT:     }
// AFFINE-NEXT:     %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x10x10x64xf32>
// AFFINE-NEXT:     affine.for %arg1 = 0 to 1 {
// AFFINE-NEXT:       affine.for %arg2 = 0 to 10 {
// AFFINE-NEXT:         affine.for %arg3 = 0 to 10 {
// AFFINE-NEXT:           affine.for %arg4 = 0 to 64 {
// AFFINE-NEXT:             affine.store %cst_2, %alloc_8[%arg1, %arg2, %arg3, %arg4] : memref<1x10x10x64xf32>
// AFFINE-NEXT:           }
// AFFINE-NEXT:         }
// AFFINE-NEXT:       }
// AFFINE-NEXT:     }
// AFFINE-NEXT:     %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x8x8x64xf32>
// AFFINE-NEXT:     affine.for %arg1 = 0 to 1 {
// AFFINE-NEXT:       affine.for %arg2 = 0 to 8 {
// AFFINE-NEXT:         affine.for %arg3 = 0 to 8 {
// AFFINE-NEXT:           affine.for %arg4 = 0 to 64 {
// AFFINE-NEXT:             affine.store %cst_2, %alloc_9[%arg1, %arg2, %arg3, %arg4] : memref<1x8x8x64xf32>
// AFFINE-NEXT:           }
// AFFINE-NEXT:         }
// AFFINE-NEXT:       }
// AFFINE-NEXT:     }
// AFFINE-NEXT:     affine.for %arg1 = 0 to 1 {
// AFFINE-NEXT:       affine.for %arg2 = 0 to 8 {
// AFFINE-NEXT:         affine.for %arg3 = 0 to 8 {
// AFFINE-NEXT:           affine.for %arg4 = 0 to 64 {
// AFFINE-NEXT:             affine.for %arg5 = 0 to 3 {
// AFFINE-NEXT:               affine.for %arg6 = 0 to 3 {
// AFFINE-NEXT:                 affine.for %arg7 = 0 to 64 {
// AFFINE-NEXT:                   %0 = affine.load %alloc_8[%arg1, %arg2 + %arg5, %arg3 + %arg6, %arg7] : memref<1x10x10x64xf32>
// AFFINE-NEXT:                   %1 = affine.load %alloc_9[%arg1, %arg2, %arg3, %arg4] : memref<1x8x8x64xf32>
// AFFINE-NEXT:                   %2 = arith.mulf %0, %cst : f32
// AFFINE-NEXT:                   %3 = arith.addf %1, %2 : f32
// AFFINE-NEXT:                   affine.store %3, %alloc_9[%arg1, %arg2, %arg3, %arg4] : memref<1x8x8x64xf32>
// AFFINE-NEXT:                 }
// AFFINE-NEXT:               }
// AFFINE-NEXT:             }
// AFFINE-NEXT:           }
// AFFINE-NEXT:         }
// AFFINE-NEXT:       }
// AFFINE-NEXT:     }
// AFFINE-NEXT:     %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x64x8x8xf32>
// AFFINE-NEXT:     affine.for %arg1 = 0 to 1 {
// AFFINE-NEXT:       affine.for %arg2 = 0 to 64 {
// AFFINE-NEXT:         affine.for %arg3 = 0 to 8 {
// AFFINE-NEXT:           affine.for %arg4 = 0 to 8 {
// AFFINE-NEXT:             %0 = affine.load %alloc_9[%arg1, %arg3, %arg4, %arg2] : memref<1x8x8x64xf32>
// AFFINE-NEXT:             affine.store %0, %alloc_10[%arg1, %arg2, %arg3, %arg4] : memref<1x64x8x8xf32>
// AFFINE-NEXT:           }
// AFFINE-NEXT:         }
// AFFINE-NEXT:       }
// AFFINE-NEXT:     }
// AFFINE-NEXT:     %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<1x64x8x8xf32>
// AFFINE-NEXT:     affine.for %arg1 = 0 to 1 {
// AFFINE-NEXT:       affine.for %arg2 = 0 to 64 {
// AFFINE-NEXT:         affine.for %arg3 = 0 to 8 {
// AFFINE-NEXT:           affine.for %arg4 = 0 to 8 {
// AFFINE-NEXT:             %0 = affine.load %alloc_10[%arg1, %arg2, %arg3, %arg4] : memref<1x64x8x8xf32>
// AFFINE-NEXT:             %1 = affine.load %arg0[%arg1, %arg2, %arg3, %arg4] : memref<1x64x8x8xf32>
// AFFINE-NEXT:             %2 = arith.addf %0, %1 : f32
// AFFINE-NEXT:             affine.store %2, %alloc_11[%arg1, %arg2, %arg3, %arg4] : memref<1x64x8x8xf32>
// AFFINE-NEXT:           }
// AFFINE-NEXT:         }
// AFFINE-NEXT:       }
// AFFINE-NEXT:     }
// AFFINE-NEXT:     %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<1x64x8x8xf32>
// AFFINE-NEXT:     affine.for %arg1 = 0 to 1 {
// AFFINE-NEXT:       affine.for %arg2 = 0 to 64 {
// AFFINE-NEXT:         affine.for %arg3 = 0 to 8 {
// AFFINE-NEXT:           affine.for %arg4 = 0 to 8 {
// AFFINE-NEXT:             %0 = affine.load %alloc_11[%arg1, %arg2, %arg3, %arg4] : memref<1x64x8x8xf32>
// AFFINE-NEXT:             %1 = arith.minimumf %0, %cst_1 : f32
// AFFINE-NEXT:             %2 = arith.maximumf %1, %cst_2 : f32
// AFFINE-NEXT:             affine.store %2, %alloc_12[%arg1, %arg2, %arg3, %arg4] : memref<1x64x8x8xf32>
// AFFINE-NEXT:           }
// AFFINE-NEXT:         }
// AFFINE-NEXT:       }
// AFFINE-NEXT:     }
// AFFINE-NEXT:     return %alloc_12 : memref<1x64x8x8xf32>
// AFFINE-NEXT:   }
// AFFINE-NEXT: }


// KERNEL:      module attributes {torch.debug_module_name = "SimpleResNetBlock"} {
// KERNEL-NEXT:   memref.global "private" constant @__constant_64xf32 : memref<64xf32> = dense<0.000000e+00> {alignment = 64 : i64}
// KERNEL-NEXT:   memref.global "private" constant @__constant_64x3x3x64xf32_0 : memref<64x3x3x64xf32> = dense<-0.0151730878> {alignment = 64 : i64}
// KERNEL-NEXT:   memref.global "private" constant @__constant_64x3x3x64xf32 : memref<64x3x3x64xf32> = dense<0.0197670367> {alignment = 64 : i64}
// KERNEL-NEXT:   func.func @forward(%arg0: memref<1x64x8x8xf32>) -> memref<1x64x8x8xf32> {
// KERNEL-NEXT:     %cst = arith.constant 0.0197670367 : f32
// KERNEL-NEXT:     %cst_0 = arith.constant -0.0151730878 : f32
// KERNEL-NEXT:     %cst_1 = arith.constant 3.40282347E+38 : f32
// KERNEL-NEXT:     %cst_2 = arith.constant 0.000000e+00 : f32
// KERNEL-NEXT:     %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x8x8x64xf32>
// KERNEL-NEXT:     %dependency_read_out, %dependency_write_out = taskflow.task @Task_0 dependency_read_in(%arg0 : memref<1x64x8x8xf32>) dependency_write_in(%alloc : memref<1x8x8x64xf32>) [original_read_memrefs(%arg0 : memref<1x64x8x8xf32>), original_write_memrefs(%alloc : memref<1x8x8x64xf32>)] : (memref<1x64x8x8xf32>, memref<1x8x8x64xf32>) -> (memref<1x64x8x8xf32>, memref<1x8x8x64xf32>) {
// KERNEL-NEXT:     ^bb0(%arg1: memref<1x64x8x8xf32>, %arg2: memref<1x8x8x64xf32>):
// KERNEL-NEXT:       %c64 = arith.constant 64 : index
// KERNEL-NEXT:       %c8 = arith.constant 8 : index
// KERNEL-NEXT:       %c0 = arith.constant 0 : index
// KERNEL-NEXT:       %c1 = arith.constant 1 : index
// KERNEL-NEXT:       %0 = taskflow.counter from %c0 to %c1 step %c1 attributes {counter_id = 0 : i32, counter_type = "root"} : index
// KERNEL-NEXT:       %1 = taskflow.counter parent(%0 : index) from %c0 to %c8 step %c1 attributes {counter_id = 1 : i32, counter_type = "relay"} : index
// KERNEL-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0 to %c8 step %c1 attributes {counter_id = 2 : i32, counter_type = "relay"} : index
// KERNEL-NEXT:       %3 = taskflow.counter parent(%2 : index) from %c0 to %c64 step %c1 attributes {counter_id = 3 : i32, counter_type = "leaf"} : index
// KERNEL-NEXT:       neura.kernel inputs(%arg1, %arg2 : memref<1x64x8x8xf32>, memref<1x8x8x64xf32>) {
// KERNEL-NEXT:       ^bb0(%arg3: memref<1x64x8x8xf32>, %arg4: memref<1x8x8x64xf32>):
// KERNEL-NEXT:         %c64_33 = arith.constant 64 : index
// KERNEL-NEXT:         %c8_34 = arith.constant 8 : index
// KERNEL-NEXT:         %c0_35 = arith.constant 0 : index
// KERNEL-NEXT:         %c1_36 = arith.constant 1 : index
// KERNEL-NEXT:         %4 = neura.counter from %c0_35 : index to %c1_36 : index step %c1_36 : index attributes {counter_id = 0 : i32, counter_type = "root"} -> index
// KERNEL-NEXT:         %5 = neura.counter from %c0_35 : index to %c8_34 : index step %c1_36 : index attributes {counter_id = 1 : i32, counter_type = "relay"} -> index
// KERNEL-NEXT:         %6 = neura.counter from %c0_35 : index to %c8_34 : index step %c1_36 : index attributes {counter_id = 2 : i32, counter_type = "relay"} -> index
// KERNEL-NEXT:         %7 = neura.counter from %c0_35 : index to %c64_33 : index step %c1_36 : index attributes {counter_id = 3 : i32, counter_type = "leaf"} -> index
// KERNEL-NEXT:         %8 = memref.load %arg3[%4, %7, %5, %6] : memref<1x64x8x8xf32>
// KERNEL-NEXT:         memref.store %8, %arg4[%4, %5, %6, %7] : memref<1x8x8x64xf32>
// KERNEL-NEXT:         neura.yield
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield reads(%arg1 : memref<1x64x8x8xf32>) writes(%arg2 : memref<1x8x8x64xf32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x10x10x64xf32>
// KERNEL-NEXT:     %dependency_write_out_4 = taskflow.task @Task_1 dependency_write_in(%alloc_3 : memref<1x10x10x64xf32>) value_inputs(%cst_2 : f32) [original_write_memrefs(%alloc_3 : memref<1x10x10x64xf32>)] : (memref<1x10x10x64xf32>, f32) -> (memref<1x10x10x64xf32>) {
// KERNEL-NEXT:     ^bb0(%arg1: memref<1x10x10x64xf32>, %arg2: f32):
// KERNEL-NEXT:       %c64 = arith.constant 64 : index
// KERNEL-NEXT:       %c10 = arith.constant 10 : index
// KERNEL-NEXT:       %c0 = arith.constant 0 : index
// KERNEL-NEXT:       %c1 = arith.constant 1 : index
// KERNEL-NEXT:       %0 = taskflow.counter from %c0 to %c1 step %c1 attributes {counter_id = 0 : i32, counter_type = "root"} : index
// KERNEL-NEXT:       %1 = taskflow.counter parent(%0 : index) from %c0 to %c10 step %c1 attributes {counter_id = 1 : i32, counter_type = "relay"} : index
// KERNEL-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0 to %c10 step %c1 attributes {counter_id = 2 : i32, counter_type = "relay"} : index
// KERNEL-NEXT:       %3 = taskflow.counter parent(%2 : index) from %c0 to %c64 step %c1 attributes {counter_id = 3 : i32, counter_type = "leaf"} : index
// KERNEL-NEXT:       neura.kernel inputs(%arg2, %arg1 : f32, memref<1x10x10x64xf32>) {
// KERNEL-NEXT:       ^bb0(%arg3: f32, %arg4: memref<1x10x10x64xf32>):
// KERNEL-NEXT:         %c64_33 = arith.constant 64 : index
// KERNEL-NEXT:         %c10_34 = arith.constant 10 : index
// KERNEL-NEXT:         %c0_35 = arith.constant 0 : index
// KERNEL-NEXT:         %c1_36 = arith.constant 1 : index
// KERNEL-NEXT:         %4 = neura.counter from %c0_35 : index to %c1_36 : index step %c1_36 : index attributes {counter_id = 0 : i32, counter_type = "root"} -> index
// KERNEL-NEXT:         %5 = neura.counter from %c0_35 : index to %c10_34 : index step %c1_36 : index attributes {counter_id = 1 : i32, counter_type = "relay"} -> index
// KERNEL-NEXT:         %6 = neura.counter from %c0_35 : index to %c10_34 : index step %c1_36 : index attributes {counter_id = 2 : i32, counter_type = "relay"} -> index
// KERNEL-NEXT:         %7 = neura.counter from %c0_35 : index to %c64_33 : index step %c1_36 : index attributes {counter_id = 3 : i32, counter_type = "leaf"} -> index
// KERNEL-NEXT:         memref.store %arg3, %arg4[%4, %5, %6, %7] : memref<1x10x10x64xf32>
// KERNEL-NEXT:         neura.yield
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield writes(%arg1 : memref<1x10x10x64xf32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x8x8x64xf32>
// KERNEL-NEXT:     %dependency_write_out_6 = taskflow.task @Task_2 dependency_write_in(%alloc_5 : memref<1x8x8x64xf32>) value_inputs(%cst_2 : f32) [original_write_memrefs(%alloc_5 : memref<1x8x8x64xf32>)] : (memref<1x8x8x64xf32>, f32) -> (memref<1x8x8x64xf32>) {
// KERNEL-NEXT:     ^bb0(%arg1: memref<1x8x8x64xf32>, %arg2: f32):
// KERNEL-NEXT:       %c64 = arith.constant 64 : index
// KERNEL-NEXT:       %c8 = arith.constant 8 : index
// KERNEL-NEXT:       %c0 = arith.constant 0 : index
// KERNEL-NEXT:       %c1 = arith.constant 1 : index
// KERNEL-NEXT:       %0 = taskflow.counter from %c0 to %c1 step %c1 attributes {counter_id = 0 : i32, counter_type = "root"} : index
// KERNEL-NEXT:       %1 = taskflow.counter parent(%0 : index) from %c0 to %c8 step %c1 attributes {counter_id = 1 : i32, counter_type = "relay"} : index
// KERNEL-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0 to %c8 step %c1 attributes {counter_id = 2 : i32, counter_type = "relay"} : index
// KERNEL-NEXT:       %3 = taskflow.counter parent(%2 : index) from %c0 to %c64 step %c1 attributes {counter_id = 3 : i32, counter_type = "leaf"} : index
// KERNEL-NEXT:       neura.kernel inputs(%arg2, %arg1 : f32, memref<1x8x8x64xf32>) {
// KERNEL-NEXT:       ^bb0(%arg3: f32, %arg4: memref<1x8x8x64xf32>):
// KERNEL-NEXT:         %c64_33 = arith.constant 64 : index
// KERNEL-NEXT:         %c8_34 = arith.constant 8 : index
// KERNEL-NEXT:         %c0_35 = arith.constant 0 : index
// KERNEL-NEXT:         %c1_36 = arith.constant 1 : index
// KERNEL-NEXT:         %4 = neura.counter from %c0_35 : index to %c1_36 : index step %c1_36 : index attributes {counter_id = 0 : i32, counter_type = "root"} -> index
// KERNEL-NEXT:         %5 = neura.counter from %c0_35 : index to %c8_34 : index step %c1_36 : index attributes {counter_id = 1 : i32, counter_type = "relay"} -> index
// KERNEL-NEXT:         %6 = neura.counter from %c0_35 : index to %c8_34 : index step %c1_36 : index attributes {counter_id = 2 : i32, counter_type = "relay"} -> index
// KERNEL-NEXT:         %7 = neura.counter from %c0_35 : index to %c64_33 : index step %c1_36 : index attributes {counter_id = 3 : i32, counter_type = "leaf"} -> index
// KERNEL-NEXT:         memref.store %arg3, %arg4[%4, %5, %6, %7] : memref<1x8x8x64xf32>
// KERNEL-NEXT:         neura.yield
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield writes(%arg1 : memref<1x8x8x64xf32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %dependency_read_out_7:2, %dependency_write_out_8 = taskflow.task @Task_3 dependency_read_in(%dependency_write_out_4, %dependency_write_out_6 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>) dependency_write_in(%dependency_write_out_6 : memref<1x8x8x64xf32>) value_inputs(%cst_0 : f32) [original_read_memrefs(%alloc_3, %alloc_5 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>), original_write_memrefs(%alloc_5 : memref<1x8x8x64xf32>)] : (memref<1x10x10x64xf32>, memref<1x8x8x64xf32>, memref<1x8x8x64xf32>, f32) -> (memref<1x10x10x64xf32>, memref<1x8x8x64xf32>, memref<1x8x8x64xf32>) {
// KERNEL-NEXT:     ^bb0(%arg1: memref<1x10x10x64xf32>, %arg2: memref<1x8x8x64xf32>, %arg3: memref<1x8x8x64xf32>, %arg4: f32):
// KERNEL-NEXT:       %c3 = arith.constant 3 : index
// KERNEL-NEXT:       %c64 = arith.constant 64 : index
// KERNEL-NEXT:       %c8 = arith.constant 8 : index
// KERNEL-NEXT:       %c0 = arith.constant 0 : index
// KERNEL-NEXT:       %c1 = arith.constant 1 : index
// KERNEL-NEXT:       %0 = taskflow.counter from %c0 to %c1 step %c1 attributes {counter_id = 0 : i32, counter_type = "root"} : index
// KERNEL-NEXT:       %1 = taskflow.counter parent(%0 : index) from %c0 to %c8 step %c1 attributes {counter_id = 1 : i32, counter_type = "relay"} : index
// KERNEL-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0 to %c8 step %c1 attributes {counter_id = 2 : i32, counter_type = "relay"} : index
// KERNEL-NEXT:       %3 = taskflow.counter parent(%2 : index) from %c0 to %c64 step %c1 attributes {counter_id = 3 : i32, counter_type = "relay"} : index
// KERNEL-NEXT:       %4 = taskflow.counter parent(%3 : index) from %c0 to %c3 step %c1 attributes {counter_id = 4 : i32, counter_type = "relay"} : index
// KERNEL-NEXT:       %5 = taskflow.counter parent(%4 : index) from %c0 to %c3 step %c1 attributes {counter_id = 5 : i32, counter_type = "relay"} : index
// KERNEL-NEXT:       %6 = taskflow.counter parent(%5 : index) from %c0 to %c64 step %c1 attributes {counter_id = 6 : i32, counter_type = "leaf"} : index
// KERNEL-NEXT:       neura.kernel inputs(%arg1, %arg3, %arg4 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>, f32) {
// KERNEL-NEXT:       ^bb0(%arg5: memref<1x10x10x64xf32>, %arg6: memref<1x8x8x64xf32>, %arg7: f32):
// KERNEL-NEXT:         %c3_33 = arith.constant 3 : index
// KERNEL-NEXT:         %c64_34 = arith.constant 64 : index
// KERNEL-NEXT:         %c8_35 = arith.constant 8 : index
// KERNEL-NEXT:         %c0_36 = arith.constant 0 : index
// KERNEL-NEXT:         %c1_37 = arith.constant 1 : index
// KERNEL-NEXT:         %7 = neura.counter from %c0_36 : index to %c1_37 : index step %c1_37 : index attributes {counter_id = 0 : i32, counter_type = "root"} -> index
// KERNEL-NEXT:         %8 = neura.counter from %c0_36 : index to %c8_35 : index step %c1_37 : index attributes {counter_id = 1 : i32, counter_type = "relay"} -> index
// KERNEL-NEXT:         %9 = neura.counter from %c0_36 : index to %c8_35 : index step %c1_37 : index attributes {counter_id = 2 : i32, counter_type = "relay"} -> index
// KERNEL-NEXT:         %10 = neura.counter from %c0_36 : index to %c64_34 : index step %c1_37 : index attributes {counter_id = 3 : i32, counter_type = "relay"} -> index
// KERNEL-NEXT:         %11 = neura.counter from %c0_36 : index to %c3_33 : index step %c1_37 : index attributes {counter_id = 4 : i32, counter_type = "relay"} -> index
// KERNEL-NEXT:         %12 = neura.counter from %c0_36 : index to %c3_33 : index step %c1_37 : index attributes {counter_id = 5 : i32, counter_type = "relay"} -> index
// KERNEL-NEXT:         %13 = neura.counter from %c0_36 : index to %c64_34 : index step %c1_37 : index attributes {counter_id = 6 : i32, counter_type = "leaf"} -> index
// KERNEL-NEXT:         %14 = arith.addi %8, %11 : index
// KERNEL-NEXT:         %15 = arith.addi %9, %12 : index
// KERNEL-NEXT:         %16 = memref.load %arg5[%7, %14, %15, %13] : memref<1x10x10x64xf32>
// KERNEL-NEXT:         %17 = memref.load %arg6[%7, %8, %9, %10] : memref<1x8x8x64xf32>
// KERNEL-NEXT:         %18 = arith.mulf %16, %arg7 : f32
// KERNEL-NEXT:         %19 = arith.addf %17, %18 : f32
// KERNEL-NEXT:         memref.store %19, %arg6[%7, %8, %9, %10] : memref<1x8x8x64xf32>
// KERNEL-NEXT:         neura.yield
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield reads(%arg1, %arg3 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>) writes(%arg3 : memref<1x8x8x64xf32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x64x8x8xf32>
// KERNEL-NEXT:     %dependency_read_out_10, %dependency_write_out_11 = taskflow.task @Task_4 dependency_read_in(%dependency_write_out_8 : memref<1x8x8x64xf32>) dependency_write_in(%alloc_9 : memref<1x64x8x8xf32>) [original_read_memrefs(%alloc_5 : memref<1x8x8x64xf32>), original_write_memrefs(%alloc_9 : memref<1x64x8x8xf32>)] : (memref<1x8x8x64xf32>, memref<1x64x8x8xf32>) -> (memref<1x8x8x64xf32>, memref<1x64x8x8xf32>) {
// KERNEL-NEXT:     ^bb0(%arg1: memref<1x8x8x64xf32>, %arg2: memref<1x64x8x8xf32>):
// KERNEL-NEXT:       %c8 = arith.constant 8 : index
// KERNEL-NEXT:       %c64 = arith.constant 64 : index
// KERNEL-NEXT:       %c0 = arith.constant 0 : index
// KERNEL-NEXT:       %c1 = arith.constant 1 : index
// KERNEL-NEXT:       %0 = taskflow.counter from %c0 to %c1 step %c1 attributes {counter_id = 0 : i32, counter_type = "root"} : index
// KERNEL-NEXT:       %1 = taskflow.counter parent(%0 : index) from %c0 to %c64 step %c1 attributes {counter_id = 1 : i32, counter_type = "relay"} : index
// KERNEL-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0 to %c8 step %c1 attributes {counter_id = 2 : i32, counter_type = "relay"} : index
// KERNEL-NEXT:       %3 = taskflow.counter parent(%2 : index) from %c0 to %c8 step %c1 attributes {counter_id = 3 : i32, counter_type = "leaf"} : index
// KERNEL-NEXT:       neura.kernel inputs(%arg1, %arg2 : memref<1x8x8x64xf32>, memref<1x64x8x8xf32>) {
// KERNEL-NEXT:       ^bb0(%arg3: memref<1x8x8x64xf32>, %arg4: memref<1x64x8x8xf32>):
// KERNEL-NEXT:         %c8_33 = arith.constant 8 : index
// KERNEL-NEXT:         %c64_34 = arith.constant 64 : index
// KERNEL-NEXT:         %c0_35 = arith.constant 0 : index
// KERNEL-NEXT:         %c1_36 = arith.constant 1 : index
// KERNEL-NEXT:         %4 = neura.counter from %c0_35 : index to %c1_36 : index step %c1_36 : index attributes {counter_id = 0 : i32, counter_type = "root"} -> index
// KERNEL-NEXT:         %5 = neura.counter from %c0_35 : index to %c64_34 : index step %c1_36 : index attributes {counter_id = 1 : i32, counter_type = "relay"} -> index
// KERNEL-NEXT:         %6 = neura.counter from %c0_35 : index to %c8_33 : index step %c1_36 : index attributes {counter_id = 2 : i32, counter_type = "relay"} -> index
// KERNEL-NEXT:         %7 = neura.counter from %c0_35 : index to %c8_33 : index step %c1_36 : index attributes {counter_id = 3 : i32, counter_type = "leaf"} -> index
// KERNEL-NEXT:         %8 = memref.load %arg3[%4, %6, %7, %5] : memref<1x8x8x64xf32>
// KERNEL-NEXT:         memref.store %8, %arg4[%4, %5, %6, %7] : memref<1x64x8x8xf32>
// KERNEL-NEXT:         neura.yield
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield reads(%arg1 : memref<1x8x8x64xf32>) writes(%arg2 : memref<1x64x8x8xf32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<1x64x8x8xf32>
// KERNEL-NEXT:     %dependency_read_out_13, %dependency_write_out_14 = taskflow.task @Task_5 dependency_read_in(%dependency_write_out_11 : memref<1x64x8x8xf32>) dependency_write_in(%alloc_12 : memref<1x64x8x8xf32>) value_inputs(%cst_1, %cst_2 : f32, f32) [original_read_memrefs(%alloc_9 : memref<1x64x8x8xf32>), original_write_memrefs(%alloc_12 : memref<1x64x8x8xf32>)] : (memref<1x64x8x8xf32>, memref<1x64x8x8xf32>, f32, f32) -> (memref<1x64x8x8xf32>, memref<1x64x8x8xf32>) {
// KERNEL-NEXT:     ^bb0(%arg1: memref<1x64x8x8xf32>, %arg2: memref<1x64x8x8xf32>, %arg3: f32, %arg4: f32):
// KERNEL-NEXT:       %c8 = arith.constant 8 : index
// KERNEL-NEXT:       %c64 = arith.constant 64 : index
// KERNEL-NEXT:       %c0 = arith.constant 0 : index
// KERNEL-NEXT:       %c1 = arith.constant 1 : index
// KERNEL-NEXT:       %0 = taskflow.counter from %c0 to %c1 step %c1 attributes {counter_id = 0 : i32, counter_type = "root"} : index
// KERNEL-NEXT:       %1 = taskflow.counter parent(%0 : index) from %c0 to %c64 step %c1 attributes {counter_id = 1 : i32, counter_type = "relay"} : index
// KERNEL-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0 to %c8 step %c1 attributes {counter_id = 2 : i32, counter_type = "relay"} : index
// KERNEL-NEXT:       %3 = taskflow.counter parent(%2 : index) from %c0 to %c8 step %c1 attributes {counter_id = 3 : i32, counter_type = "leaf"} : index
// KERNEL-NEXT:       neura.kernel inputs(%arg1, %arg3, %arg4, %arg2 : memref<1x64x8x8xf32>, f32, f32, memref<1x64x8x8xf32>) {
// KERNEL-NEXT:       ^bb0(%arg5: memref<1x64x8x8xf32>, %arg6: f32, %arg7: f32, %arg8: memref<1x64x8x8xf32>):
// KERNEL-NEXT:         %c8_33 = arith.constant 8 : index
// KERNEL-NEXT:         %c64_34 = arith.constant 64 : index
// KERNEL-NEXT:         %c0_35 = arith.constant 0 : index
// KERNEL-NEXT:         %c1_36 = arith.constant 1 : index
// KERNEL-NEXT:         %4 = neura.counter from %c0_35 : index to %c1_36 : index step %c1_36 : index attributes {counter_id = 0 : i32, counter_type = "root"} -> index
// KERNEL-NEXT:         %5 = neura.counter from %c0_35 : index to %c64_34 : index step %c1_36 : index attributes {counter_id = 1 : i32, counter_type = "relay"} -> index
// KERNEL-NEXT:         %6 = neura.counter from %c0_35 : index to %c8_33 : index step %c1_36 : index attributes {counter_id = 2 : i32, counter_type = "relay"} -> index
// KERNEL-NEXT:         %7 = neura.counter from %c0_35 : index to %c8_33 : index step %c1_36 : index attributes {counter_id = 3 : i32, counter_type = "leaf"} -> index
// KERNEL-NEXT:         %8 = memref.load %arg5[%4, %5, %6, %7] : memref<1x64x8x8xf32>
// KERNEL-NEXT:         %9 = arith.minimumf %8, %arg6 : f32
// KERNEL-NEXT:         %10 = arith.maximumf %9, %arg7 : f32
// KERNEL-NEXT:         memref.store %10, %arg8[%4, %5, %6, %7] : memref<1x64x8x8xf32>
// KERNEL-NEXT:         neura.yield
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield reads(%arg1 : memref<1x64x8x8xf32>) writes(%arg2 : memref<1x64x8x8xf32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<1x8x8x64xf32>
// KERNEL-NEXT:     %dependency_read_out_16, %dependency_write_out_17 = taskflow.task @Task_6 dependency_read_in(%dependency_write_out_14 : memref<1x64x8x8xf32>) dependency_write_in(%alloc_15 : memref<1x8x8x64xf32>) [original_read_memrefs(%alloc_12 : memref<1x64x8x8xf32>), original_write_memrefs(%alloc_15 : memref<1x8x8x64xf32>)] : (memref<1x64x8x8xf32>, memref<1x8x8x64xf32>) -> (memref<1x64x8x8xf32>, memref<1x8x8x64xf32>) {
// KERNEL-NEXT:     ^bb0(%arg1: memref<1x64x8x8xf32>, %arg2: memref<1x8x8x64xf32>):
// KERNEL-NEXT:       %c64 = arith.constant 64 : index
// KERNEL-NEXT:       %c8 = arith.constant 8 : index
// KERNEL-NEXT:       %c0 = arith.constant 0 : index
// KERNEL-NEXT:       %c1 = arith.constant 1 : index
// KERNEL-NEXT:       %0 = taskflow.counter from %c0 to %c1 step %c1 attributes {counter_id = 0 : i32, counter_type = "root"} : index
// KERNEL-NEXT:       %1 = taskflow.counter parent(%0 : index) from %c0 to %c8 step %c1 attributes {counter_id = 1 : i32, counter_type = "relay"} : index
// KERNEL-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0 to %c8 step %c1 attributes {counter_id = 2 : i32, counter_type = "relay"} : index
// KERNEL-NEXT:       %3 = taskflow.counter parent(%2 : index) from %c0 to %c64 step %c1 attributes {counter_id = 3 : i32, counter_type = "leaf"} : index
// KERNEL-NEXT:       neura.kernel inputs(%arg1, %arg2 : memref<1x64x8x8xf32>, memref<1x8x8x64xf32>) {
// KERNEL-NEXT:       ^bb0(%arg3: memref<1x64x8x8xf32>, %arg4: memref<1x8x8x64xf32>):
// KERNEL-NEXT:         %c64_33 = arith.constant 64 : index
// KERNEL-NEXT:         %c8_34 = arith.constant 8 : index
// KERNEL-NEXT:         %c0_35 = arith.constant 0 : index
// KERNEL-NEXT:         %c1_36 = arith.constant 1 : index
// KERNEL-NEXT:         %4 = neura.counter from %c0_35 : index to %c1_36 : index step %c1_36 : index attributes {counter_id = 0 : i32, counter_type = "root"} -> index
// KERNEL-NEXT:         %5 = neura.counter from %c0_35 : index to %c8_34 : index step %c1_36 : index attributes {counter_id = 1 : i32, counter_type = "relay"} -> index
// KERNEL-NEXT:         %6 = neura.counter from %c0_35 : index to %c8_34 : index step %c1_36 : index attributes {counter_id = 2 : i32, counter_type = "relay"} -> index
// KERNEL-NEXT:         %7 = neura.counter from %c0_35 : index to %c64_33 : index step %c1_36 : index attributes {counter_id = 3 : i32, counter_type = "leaf"} -> index
// KERNEL-NEXT:         %8 = memref.load %arg3[%4, %7, %5, %6] : memref<1x64x8x8xf32>
// KERNEL-NEXT:         memref.store %8, %arg4[%4, %5, %6, %7] : memref<1x8x8x64xf32>
// KERNEL-NEXT:         neura.yield
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield reads(%arg1 : memref<1x64x8x8xf32>) writes(%arg2 : memref<1x8x8x64xf32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %alloc_18 = memref.alloc() {alignment = 64 : i64} : memref<1x10x10x64xf32>
// KERNEL-NEXT:     %dependency_write_out_19 = taskflow.task @Task_7 dependency_write_in(%alloc_18 : memref<1x10x10x64xf32>) value_inputs(%cst_2 : f32) [original_write_memrefs(%alloc_18 : memref<1x10x10x64xf32>)] : (memref<1x10x10x64xf32>, f32) -> (memref<1x10x10x64xf32>) {
// KERNEL-NEXT:     ^bb0(%arg1: memref<1x10x10x64xf32>, %arg2: f32):
// KERNEL-NEXT:       %c64 = arith.constant 64 : index
// KERNEL-NEXT:       %c10 = arith.constant 10 : index
// KERNEL-NEXT:       %c0 = arith.constant 0 : index
// KERNEL-NEXT:       %c1 = arith.constant 1 : index
// KERNEL-NEXT:       %0 = taskflow.counter from %c0 to %c1 step %c1 attributes {counter_id = 0 : i32, counter_type = "root"} : index
// KERNEL-NEXT:       %1 = taskflow.counter parent(%0 : index) from %c0 to %c10 step %c1 attributes {counter_id = 1 : i32, counter_type = "relay"} : index
// KERNEL-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0 to %c10 step %c1 attributes {counter_id = 2 : i32, counter_type = "relay"} : index
// KERNEL-NEXT:       %3 = taskflow.counter parent(%2 : index) from %c0 to %c64 step %c1 attributes {counter_id = 3 : i32, counter_type = "leaf"} : index
// KERNEL-NEXT:       neura.kernel inputs(%arg2, %arg1 : f32, memref<1x10x10x64xf32>) {
// KERNEL-NEXT:       ^bb0(%arg3: f32, %arg4: memref<1x10x10x64xf32>):
// KERNEL-NEXT:         %c64_33 = arith.constant 64 : index
// KERNEL-NEXT:         %c10_34 = arith.constant 10 : index
// KERNEL-NEXT:         %c0_35 = arith.constant 0 : index
// KERNEL-NEXT:         %c1_36 = arith.constant 1 : index
// KERNEL-NEXT:         %4 = neura.counter from %c0_35 : index to %c1_36 : index step %c1_36 : index attributes {counter_id = 0 : i32, counter_type = "root"} -> index
// KERNEL-NEXT:         %5 = neura.counter from %c0_35 : index to %c10_34 : index step %c1_36 : index attributes {counter_id = 1 : i32, counter_type = "relay"} -> index
// KERNEL-NEXT:         %6 = neura.counter from %c0_35 : index to %c10_34 : index step %c1_36 : index attributes {counter_id = 2 : i32, counter_type = "relay"} -> index
// KERNEL-NEXT:         %7 = neura.counter from %c0_35 : index to %c64_33 : index step %c1_36 : index attributes {counter_id = 3 : i32, counter_type = "leaf"} -> index
// KERNEL-NEXT:         memref.store %arg3, %arg4[%4, %5, %6, %7] : memref<1x10x10x64xf32>
// KERNEL-NEXT:         neura.yield
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield writes(%arg1 : memref<1x10x10x64xf32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %alloc_20 = memref.alloc() {alignment = 64 : i64} : memref<1x8x8x64xf32>
// KERNEL-NEXT:     %dependency_write_out_21 = taskflow.task @Task_8 dependency_write_in(%alloc_20 : memref<1x8x8x64xf32>) value_inputs(%cst_2 : f32) [original_write_memrefs(%alloc_20 : memref<1x8x8x64xf32>)] : (memref<1x8x8x64xf32>, f32) -> (memref<1x8x8x64xf32>) {
// KERNEL-NEXT:     ^bb0(%arg1: memref<1x8x8x64xf32>, %arg2: f32):
// KERNEL-NEXT:       %c64 = arith.constant 64 : index
// KERNEL-NEXT:       %c8 = arith.constant 8 : index
// KERNEL-NEXT:       %c0 = arith.constant 0 : index
// KERNEL-NEXT:       %c1 = arith.constant 1 : index
// KERNEL-NEXT:       %0 = taskflow.counter from %c0 to %c1 step %c1 attributes {counter_id = 0 : i32, counter_type = "root"} : index
// KERNEL-NEXT:       %1 = taskflow.counter parent(%0 : index) from %c0 to %c8 step %c1 attributes {counter_id = 1 : i32, counter_type = "relay"} : index
// KERNEL-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0 to %c8 step %c1 attributes {counter_id = 2 : i32, counter_type = "relay"} : index
// KERNEL-NEXT:       %3 = taskflow.counter parent(%2 : index) from %c0 to %c64 step %c1 attributes {counter_id = 3 : i32, counter_type = "leaf"} : index
// KERNEL-NEXT:       neura.kernel inputs(%arg2, %arg1 : f32, memref<1x8x8x64xf32>) {
// KERNEL-NEXT:       ^bb0(%arg3: f32, %arg4: memref<1x8x8x64xf32>):
// KERNEL-NEXT:         %c64_33 = arith.constant 64 : index
// KERNEL-NEXT:         %c8_34 = arith.constant 8 : index
// KERNEL-NEXT:         %c0_35 = arith.constant 0 : index
// KERNEL-NEXT:         %c1_36 = arith.constant 1 : index
// KERNEL-NEXT:         %4 = neura.counter from %c0_35 : index to %c1_36 : index step %c1_36 : index attributes {counter_id = 0 : i32, counter_type = "root"} -> index
// KERNEL-NEXT:         %5 = neura.counter from %c0_35 : index to %c8_34 : index step %c1_36 : index attributes {counter_id = 1 : i32, counter_type = "relay"} -> index
// KERNEL-NEXT:         %6 = neura.counter from %c0_35 : index to %c8_34 : index step %c1_36 : index attributes {counter_id = 2 : i32, counter_type = "relay"} -> index
// KERNEL-NEXT:         %7 = neura.counter from %c0_35 : index to %c64_33 : index step %c1_36 : index attributes {counter_id = 3 : i32, counter_type = "leaf"} -> index
// KERNEL-NEXT:         memref.store %arg3, %arg4[%4, %5, %6, %7] : memref<1x8x8x64xf32>
// KERNEL-NEXT:         neura.yield
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield writes(%arg1 : memref<1x8x8x64xf32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %dependency_read_out_22:2, %dependency_write_out_23 = taskflow.task @Task_9 dependency_read_in(%dependency_write_out_19, %dependency_write_out_21 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>) dependency_write_in(%dependency_write_out_21 : memref<1x8x8x64xf32>) value_inputs(%cst : f32) [original_read_memrefs(%alloc_18, %alloc_20 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>), original_write_memrefs(%alloc_20 : memref<1x8x8x64xf32>)] : (memref<1x10x10x64xf32>, memref<1x8x8x64xf32>, memref<1x8x8x64xf32>, f32) -> (memref<1x10x10x64xf32>, memref<1x8x8x64xf32>, memref<1x8x8x64xf32>) {
// KERNEL-NEXT:     ^bb0(%arg1: memref<1x10x10x64xf32>, %arg2: memref<1x8x8x64xf32>, %arg3: memref<1x8x8x64xf32>, %arg4: f32):
// KERNEL-NEXT:       %c3 = arith.constant 3 : index
// KERNEL-NEXT:       %c64 = arith.constant 64 : index
// KERNEL-NEXT:       %c8 = arith.constant 8 : index
// KERNEL-NEXT:       %c0 = arith.constant 0 : index
// KERNEL-NEXT:       %c1 = arith.constant 1 : index
// KERNEL-NEXT:       %0 = taskflow.counter from %c0 to %c1 step %c1 attributes {counter_id = 0 : i32, counter_type = "root"} : index
// KERNEL-NEXT:       %1 = taskflow.counter parent(%0 : index) from %c0 to %c8 step %c1 attributes {counter_id = 1 : i32, counter_type = "relay"} : index
// KERNEL-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0 to %c8 step %c1 attributes {counter_id = 2 : i32, counter_type = "relay"} : index
// KERNEL-NEXT:       %3 = taskflow.counter parent(%2 : index) from %c0 to %c64 step %c1 attributes {counter_id = 3 : i32, counter_type = "relay"} : index
// KERNEL-NEXT:       %4 = taskflow.counter parent(%3 : index) from %c0 to %c3 step %c1 attributes {counter_id = 4 : i32, counter_type = "relay"} : index
// KERNEL-NEXT:       %5 = taskflow.counter parent(%4 : index) from %c0 to %c3 step %c1 attributes {counter_id = 5 : i32, counter_type = "relay"} : index
// KERNEL-NEXT:       %6 = taskflow.counter parent(%5 : index) from %c0 to %c64 step %c1 attributes {counter_id = 6 : i32, counter_type = "leaf"} : index
// KERNEL-NEXT:       neura.kernel inputs(%arg1, %arg3, %arg4 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>, f32) {
// KERNEL-NEXT:       ^bb0(%arg5: memref<1x10x10x64xf32>, %arg6: memref<1x8x8x64xf32>, %arg7: f32):
// KERNEL-NEXT:         %c3_33 = arith.constant 3 : index
// KERNEL-NEXT:         %c64_34 = arith.constant 64 : index
// KERNEL-NEXT:         %c8_35 = arith.constant 8 : index
// KERNEL-NEXT:         %c0_36 = arith.constant 0 : index
// KERNEL-NEXT:         %c1_37 = arith.constant 1 : index
// KERNEL-NEXT:         %7 = neura.counter from %c0_36 : index to %c1_37 : index step %c1_37 : index attributes {counter_id = 0 : i32, counter_type = "root"} -> index
// KERNEL-NEXT:         %8 = neura.counter from %c0_36 : index to %c8_35 : index step %c1_37 : index attributes {counter_id = 1 : i32, counter_type = "relay"} -> index
// KERNEL-NEXT:         %9 = neura.counter from %c0_36 : index to %c8_35 : index step %c1_37 : index attributes {counter_id = 2 : i32, counter_type = "relay"} -> index
// KERNEL-NEXT:         %10 = neura.counter from %c0_36 : index to %c64_34 : index step %c1_37 : index attributes {counter_id = 3 : i32, counter_type = "relay"} -> index
// KERNEL-NEXT:         %11 = neura.counter from %c0_36 : index to %c3_33 : index step %c1_37 : index attributes {counter_id = 4 : i32, counter_type = "relay"} -> index
// KERNEL-NEXT:         %12 = neura.counter from %c0_36 : index to %c3_33 : index step %c1_37 : index attributes {counter_id = 5 : i32, counter_type = "relay"} -> index
// KERNEL-NEXT:         %13 = neura.counter from %c0_36 : index to %c64_34 : index step %c1_37 : index attributes {counter_id = 6 : i32, counter_type = "leaf"} -> index
// KERNEL-NEXT:         %14 = arith.addi %8, %11 : index
// KERNEL-NEXT:         %15 = arith.addi %9, %12 : index
// KERNEL-NEXT:         %16 = memref.load %arg5[%7, %14, %15, %13] : memref<1x10x10x64xf32>
// KERNEL-NEXT:         %17 = memref.load %arg6[%7, %8, %9, %10] : memref<1x8x8x64xf32>
// KERNEL-NEXT:         %18 = arith.mulf %16, %arg7 : f32
// KERNEL-NEXT:         %19 = arith.addf %17, %18 : f32
// KERNEL-NEXT:         memref.store %19, %arg6[%7, %8, %9, %10] : memref<1x8x8x64xf32>
// KERNEL-NEXT:         neura.yield
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield reads(%arg1, %arg3 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>) writes(%arg3 : memref<1x8x8x64xf32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %alloc_24 = memref.alloc() {alignment = 64 : i64} : memref<1x64x8x8xf32>
// KERNEL-NEXT:     %dependency_read_out_25, %dependency_write_out_26 = taskflow.task @Task_10 dependency_read_in(%dependency_write_out_23 : memref<1x8x8x64xf32>) dependency_write_in(%alloc_24 : memref<1x64x8x8xf32>) [original_read_memrefs(%alloc_20 : memref<1x8x8x64xf32>), original_write_memrefs(%alloc_24 : memref<1x64x8x8xf32>)] : (memref<1x8x8x64xf32>, memref<1x64x8x8xf32>) -> (memref<1x8x8x64xf32>, memref<1x64x8x8xf32>) {
// KERNEL-NEXT:     ^bb0(%arg1: memref<1x8x8x64xf32>, %arg2: memref<1x64x8x8xf32>):
// KERNEL-NEXT:       %c8 = arith.constant 8 : index
// KERNEL-NEXT:       %c64 = arith.constant 64 : index
// KERNEL-NEXT:       %c0 = arith.constant 0 : index
// KERNEL-NEXT:       %c1 = arith.constant 1 : index
// KERNEL-NEXT:       %0 = taskflow.counter from %c0 to %c1 step %c1 attributes {counter_id = 0 : i32, counter_type = "root"} : index
// KERNEL-NEXT:       %1 = taskflow.counter parent(%0 : index) from %c0 to %c64 step %c1 attributes {counter_id = 1 : i32, counter_type = "relay"} : index
// KERNEL-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0 to %c8 step %c1 attributes {counter_id = 2 : i32, counter_type = "relay"} : index
// KERNEL-NEXT:       %3 = taskflow.counter parent(%2 : index) from %c0 to %c8 step %c1 attributes {counter_id = 3 : i32, counter_type = "leaf"} : index
// KERNEL-NEXT:       neura.kernel inputs(%arg1, %arg2 : memref<1x8x8x64xf32>, memref<1x64x8x8xf32>) {
// KERNEL-NEXT:       ^bb0(%arg3: memref<1x8x8x64xf32>, %arg4: memref<1x64x8x8xf32>):
// KERNEL-NEXT:         %c8_33 = arith.constant 8 : index
// KERNEL-NEXT:         %c64_34 = arith.constant 64 : index
// KERNEL-NEXT:         %c0_35 = arith.constant 0 : index
// KERNEL-NEXT:         %c1_36 = arith.constant 1 : index
// KERNEL-NEXT:         %4 = neura.counter from %c0_35 : index to %c1_36 : index step %c1_36 : index attributes {counter_id = 0 : i32, counter_type = "root"} -> index
// KERNEL-NEXT:         %5 = neura.counter from %c0_35 : index to %c64_34 : index step %c1_36 : index attributes {counter_id = 1 : i32, counter_type = "relay"} -> index
// KERNEL-NEXT:         %6 = neura.counter from %c0_35 : index to %c8_33 : index step %c1_36 : index attributes {counter_id = 2 : i32, counter_type = "relay"} -> index
// KERNEL-NEXT:         %7 = neura.counter from %c0_35 : index to %c8_33 : index step %c1_36 : index attributes {counter_id = 3 : i32, counter_type = "leaf"} -> index
// KERNEL-NEXT:         %8 = memref.load %arg3[%4, %6, %7, %5] : memref<1x8x8x64xf32>
// KERNEL-NEXT:         memref.store %8, %arg4[%4, %5, %6, %7] : memref<1x64x8x8xf32>
// KERNEL-NEXT:         neura.yield
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield reads(%arg1 : memref<1x8x8x64xf32>) writes(%arg2 : memref<1x64x8x8xf32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %alloc_27 = memref.alloc() {alignment = 64 : i64} : memref<1x64x8x8xf32>
// KERNEL-NEXT:     %dependency_read_out_28:2, %dependency_write_out_29 = taskflow.task @Task_11 dependency_read_in(%dependency_write_out_26, %dependency_read_out : memref<1x64x8x8xf32>, memref<1x64x8x8xf32>) dependency_write_in(%alloc_27 : memref<1x64x8x8xf32>) [original_read_memrefs(%alloc_24, %arg0 : memref<1x64x8x8xf32>, memref<1x64x8x8xf32>), original_write_memrefs(%alloc_27 : memref<1x64x8x8xf32>)] : (memref<1x64x8x8xf32>, memref<1x64x8x8xf32>, memref<1x64x8x8xf32>) -> (memref<1x64x8x8xf32>, memref<1x64x8x8xf32>, memref<1x64x8x8xf32>) {
// KERNEL-NEXT:     ^bb0(%arg1: memref<1x64x8x8xf32>, %arg2: memref<1x64x8x8xf32>, %arg3: memref<1x64x8x8xf32>):
// KERNEL-NEXT:       %c8 = arith.constant 8 : index
// KERNEL-NEXT:       %c64 = arith.constant 64 : index
// KERNEL-NEXT:       %c0 = arith.constant 0 : index
// KERNEL-NEXT:       %c1 = arith.constant 1 : index
// KERNEL-NEXT:       %0 = taskflow.counter from %c0 to %c1 step %c1 attributes {counter_id = 0 : i32, counter_type = "root"} : index
// KERNEL-NEXT:       %1 = taskflow.counter parent(%0 : index) from %c0 to %c64 step %c1 attributes {counter_id = 1 : i32, counter_type = "relay"} : index
// KERNEL-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0 to %c8 step %c1 attributes {counter_id = 2 : i32, counter_type = "relay"} : index
// KERNEL-NEXT:       %3 = taskflow.counter parent(%2 : index) from %c0 to %c8 step %c1 attributes {counter_id = 3 : i32, counter_type = "leaf"} : index
// KERNEL-NEXT:       neura.kernel inputs(%arg1, %arg2, %arg3 : memref<1x64x8x8xf32>, memref<1x64x8x8xf32>, memref<1x64x8x8xf32>) {
// KERNEL-NEXT:       ^bb0(%arg4: memref<1x64x8x8xf32>, %arg5: memref<1x64x8x8xf32>, %arg6: memref<1x64x8x8xf32>):
// KERNEL-NEXT:         %c8_33 = arith.constant 8 : index
// KERNEL-NEXT:         %c64_34 = arith.constant 64 : index
// KERNEL-NEXT:         %c0_35 = arith.constant 0 : index
// KERNEL-NEXT:         %c1_36 = arith.constant 1 : index
// KERNEL-NEXT:         %4 = neura.counter from %c0_35 : index to %c1_36 : index step %c1_36 : index attributes {counter_id = 0 : i32, counter_type = "root"} -> index
// KERNEL-NEXT:         %5 = neura.counter from %c0_35 : index to %c64_34 : index step %c1_36 : index attributes {counter_id = 1 : i32, counter_type = "relay"} -> index
// KERNEL-NEXT:         %6 = neura.counter from %c0_35 : index to %c8_33 : index step %c1_36 : index attributes {counter_id = 2 : i32, counter_type = "relay"} -> index
// KERNEL-NEXT:         %7 = neura.counter from %c0_35 : index to %c8_33 : index step %c1_36 : index attributes {counter_id = 3 : i32, counter_type = "leaf"} -> index
// KERNEL-NEXT:         %8 = memref.load %arg4[%4, %5, %6, %7] : memref<1x64x8x8xf32>
// KERNEL-NEXT:         %9 = memref.load %arg5[%4, %5, %6, %7] : memref<1x64x8x8xf32>
// KERNEL-NEXT:         %10 = arith.addf %8, %9 : f32
// KERNEL-NEXT:         memref.store %10, %arg6[%4, %5, %6, %7] : memref<1x64x8x8xf32>
// KERNEL-NEXT:         neura.yield
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield reads(%arg1, %arg2 : memref<1x64x8x8xf32>, memref<1x64x8x8xf32>) writes(%arg3 : memref<1x64x8x8xf32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %alloc_30 = memref.alloc() {alignment = 64 : i64} : memref<1x64x8x8xf32>
// KERNEL-NEXT:     %dependency_read_out_31, %dependency_write_out_32 = taskflow.task @Task_12 dependency_read_in(%dependency_write_out_29 : memref<1x64x8x8xf32>) dependency_write_in(%alloc_30 : memref<1x64x8x8xf32>) value_inputs(%cst_1, %cst_2 : f32, f32) [original_read_memrefs(%alloc_27 : memref<1x64x8x8xf32>), original_write_memrefs(%alloc_30 : memref<1x64x8x8xf32>)] : (memref<1x64x8x8xf32>, memref<1x64x8x8xf32>, f32, f32) -> (memref<1x64x8x8xf32>, memref<1x64x8x8xf32>) {
// KERNEL-NEXT:     ^bb0(%arg1: memref<1x64x8x8xf32>, %arg2: memref<1x64x8x8xf32>, %arg3: f32, %arg4: f32):
// KERNEL-NEXT:       %c8 = arith.constant 8 : index
// KERNEL-NEXT:       %c64 = arith.constant 64 : index
// KERNEL-NEXT:       %c0 = arith.constant 0 : index
// KERNEL-NEXT:       %c1 = arith.constant 1 : index
// KERNEL-NEXT:       %0 = taskflow.counter from %c0 to %c1 step %c1 attributes {counter_id = 0 : i32, counter_type = "root"} : index
// KERNEL-NEXT:       %1 = taskflow.counter parent(%0 : index) from %c0 to %c64 step %c1 attributes {counter_id = 1 : i32, counter_type = "relay"} : index
// KERNEL-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0 to %c8 step %c1 attributes {counter_id = 2 : i32, counter_type = "relay"} : index
// KERNEL-NEXT:       %3 = taskflow.counter parent(%2 : index) from %c0 to %c8 step %c1 attributes {counter_id = 3 : i32, counter_type = "leaf"} : index
// KERNEL-NEXT:       neura.kernel inputs(%arg1, %arg3, %arg4, %arg2 : memref<1x64x8x8xf32>, f32, f32, memref<1x64x8x8xf32>) {
// KERNEL-NEXT:       ^bb0(%arg5: memref<1x64x8x8xf32>, %arg6: f32, %arg7: f32, %arg8: memref<1x64x8x8xf32>):
// KERNEL-NEXT:         %c8_33 = arith.constant 8 : index
// KERNEL-NEXT:         %c64_34 = arith.constant 64 : index
// KERNEL-NEXT:         %c0_35 = arith.constant 0 : index
// KERNEL-NEXT:         %c1_36 = arith.constant 1 : index
// KERNEL-NEXT:         %4 = neura.counter from %c0_35 : index to %c1_36 : index step %c1_36 : index attributes {counter_id = 0 : i32, counter_type = "root"} -> index
// KERNEL-NEXT:         %5 = neura.counter from %c0_35 : index to %c64_34 : index step %c1_36 : index attributes {counter_id = 1 : i32, counter_type = "relay"} -> index
// KERNEL-NEXT:         %6 = neura.counter from %c0_35 : index to %c8_33 : index step %c1_36 : index attributes {counter_id = 2 : i32, counter_type = "relay"} -> index
// KERNEL-NEXT:         %7 = neura.counter from %c0_35 : index to %c8_33 : index step %c1_36 : index attributes {counter_id = 3 : i32, counter_type = "leaf"} -> index
// KERNEL-NEXT:         %8 = memref.load %arg5[%4, %5, %6, %7] : memref<1x64x8x8xf32>
// KERNEL-NEXT:         %9 = arith.minimumf %8, %arg6 : f32
// KERNEL-NEXT:         %10 = arith.maximumf %9, %arg7 : f32
// KERNEL-NEXT:         memref.store %10, %arg8[%4, %5, %6, %7] : memref<1x64x8x8xf32>
// KERNEL-NEXT:         neura.yield
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield reads(%arg1 : memref<1x64x8x8xf32>) writes(%arg2 : memref<1x64x8x8xf32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     return %dependency_write_out_32 : memref<1x64x8x8xf32>
// KERNEL-NEXT:   }
// KERNEL-NEXT: }



// STREAM: module attributes {torch.debug_module_name = "SimpleResNetBlock"} {
// STREAM-NEXT:   memref.global "private" constant @__constant_64xf32 : memref<64xf32> = dense<0.000000e+00> {alignment = 64 : i64}
// STREAM-NEXT:   memref.global "private" constant @__constant_64x3x3x64xf32_0 : memref<64x3x3x64xf32> = dense<-0.0151730878> {alignment = 64 : i64}
// STREAM-NEXT:   memref.global "private" constant @__constant_64x3x3x64xf32 : memref<64x3x3x64xf32> = dense<0.0197670367> {alignment = 64 : i64}
// STREAM-NEXT:   func.func @forward(%arg0: memref<1x64x8x8xf32>) -> memref<1x64x8x8xf32> {
// STREAM-NEXT:     %cst = arith.constant 0.0197670367 : f32
// STREAM-NEXT:     %cst_0 = arith.constant -0.0151730878 : f32
// STREAM-NEXT:     %cst_1 = arith.constant 3.40282347E+38 : f32
// STREAM-NEXT:     %cst_2 = arith.constant 0.000000e+00 : f32
// STREAM-NEXT:     %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x8x8x64xf32>
// STREAM-NEXT:     %dependency_read_out, %dependency_write_out = taskflow.task @Task_0 dependency_read_in(%arg0 : memref<1x64x8x8xf32>) dependency_write_in(%alloc : memref<1x8x8x64xf32>) [original_read_memrefs(%arg0 : memref<1x64x8x8xf32>), original_write_memrefs(%alloc : memref<1x8x8x64xf32>)] : (memref<1x64x8x8xf32>, memref<1x8x8x64xf32>) -> (memref<1x64x8x8xf32>, memref<1x8x8x64xf32>) {
// STREAM-NEXT:     ^bb0(%arg1: memref<1x64x8x8xf32>, %arg2: memref<1x8x8x64xf32>):
// STREAM-NEXT:       affine.for %arg3 = 0 to 1 {
// STREAM-NEXT:         affine.for %arg4 = 0 to 8 {
// STREAM-NEXT:           affine.for %arg5 = 0 to 8 {
// STREAM-NEXT:             affine.for %arg6 = 0 to 64 {
// STREAM-NEXT:               %0 = affine.load %arg1[%arg3, %arg6, %arg4, %arg5] : memref<1x64x8x8xf32>
// STREAM-NEXT:               affine.store %0, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<1x8x8x64xf32>
// STREAM-NEXT:             }
// STREAM-NEXT:           }
// STREAM-NEXT:         }
// STREAM-NEXT:       }
// STREAM-NEXT:       taskflow.yield reads(%arg1 : memref<1x64x8x8xf32>) writes(%arg2 : memref<1x8x8x64xf32>)
// STREAM-NEXT:     }
// STREAM-NEXT:     %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x10x10x64xf32>
// STREAM-NEXT:     %dependency_write_out_4 = taskflow.task @Task_1 dependency_write_in(%alloc_3 : memref<1x10x10x64xf32>) value_inputs(%cst_2 : f32) [original_write_memrefs(%alloc_3 : memref<1x10x10x64xf32>)] : (memref<1x10x10x64xf32>, f32) -> (memref<1x10x10x64xf32>) {
// STREAM-NEXT:     ^bb0(%arg1: memref<1x10x10x64xf32>, %arg2: f32):
// STREAM-NEXT:       affine.for %arg3 = 0 to 1 {
// STREAM-NEXT:         affine.for %arg4 = 0 to 10 {
// STREAM-NEXT:           affine.for %arg5 = 0 to 10 {
// STREAM-NEXT:             affine.for %arg6 = 0 to 64 {
// STREAM-NEXT:               affine.store %arg2, %arg1[%arg3, %arg4, %arg5, %arg6] : memref<1x10x10x64xf32>
// STREAM-NEXT:             }
// STREAM-NEXT:           }
// STREAM-NEXT:         }
// STREAM-NEXT:       }
// STREAM-NEXT:       taskflow.yield writes(%arg1 : memref<1x10x10x64xf32>)
// STREAM-NEXT:     }
// STREAM-NEXT:     %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x8x8x64xf32>
// STREAM-NEXT:     %dependency_write_out_6 = taskflow.task @Task_2 dependency_write_in(%alloc_5 : memref<1x8x8x64xf32>) value_inputs(%cst_2 : f32) [original_write_memrefs(%alloc_5 : memref<1x8x8x64xf32>)] : (memref<1x8x8x64xf32>, f32) -> (memref<1x8x8x64xf32>) {
// STREAM-NEXT:     ^bb0(%arg1: memref<1x8x8x64xf32>, %arg2: f32):
// STREAM-NEXT:       affine.for %arg3 = 0 to 1 {
// STREAM-NEXT:         affine.for %arg4 = 0 to 8 {
// STREAM-NEXT:           affine.for %arg5 = 0 to 8 {
// STREAM-NEXT:             affine.for %arg6 = 0 to 64 {
// STREAM-NEXT:               affine.store %arg2, %arg1[%arg3, %arg4, %arg5, %arg6] : memref<1x8x8x64xf32>
// STREAM-NEXT:             }
// STREAM-NEXT:           }
// STREAM-NEXT:         }
// STREAM-NEXT:       }
// STREAM-NEXT:       taskflow.yield writes(%arg1 : memref<1x8x8x64xf32>)
// STREAM-NEXT:     }
// STREAM-NEXT:     %dependency_read_out_7:2, %dependency_write_out_8 = taskflow.task @Task_3 dependency_read_in(%dependency_write_out_4, %dependency_write_out_6 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>) dependency_write_in(%dependency_write_out_6 : memref<1x8x8x64xf32>) value_inputs(%cst_0 : f32) [original_read_memrefs(%alloc_3, %alloc_5 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>), original_write_memrefs(%alloc_5 : memref<1x8x8x64xf32>)] : (memref<1x10x10x64xf32>, memref<1x8x8x64xf32>, memref<1x8x8x64xf32>, f32) -> (memref<1x10x10x64xf32>, memref<1x8x8x64xf32>, memref<1x8x8x64xf32>) {
// STREAM-NEXT:     ^bb0(%arg1: memref<1x10x10x64xf32>, %arg2: memref<1x8x8x64xf32>, %arg3: memref<1x8x8x64xf32>, %arg4: f32):
// STREAM-NEXT:       affine.for %arg5 = 0 to 1 {
// STREAM-NEXT:         affine.for %arg6 = 0 to 8 {
// STREAM-NEXT:           affine.for %arg7 = 0 to 8 {
// STREAM-NEXT:             affine.for %arg8 = 0 to 64 {
// STREAM-NEXT:               affine.for %arg9 = 0 to 3 {
// STREAM-NEXT:                 affine.for %arg10 = 0 to 3 {
// STREAM-NEXT:                   affine.for %arg11 = 0 to 64 {
// STREAM-NEXT:                     %0 = affine.load %arg1[%arg5, %arg6 + %arg9, %arg7 + %arg10, %arg11] : memref<1x10x10x64xf32>
// STREAM-NEXT:                     %1 = affine.load %arg3[%arg5, %arg6, %arg7, %arg8] : memref<1x8x8x64xf32>
// STREAM-NEXT:                     %2 = arith.mulf %0, %arg4 : f32
// STREAM-NEXT:                     %3 = arith.addf %1, %2 : f32
// STREAM-NEXT:                     affine.store %3, %arg3[%arg5, %arg6, %arg7, %arg8] : memref<1x8x8x64xf32>
// STREAM-NEXT:                   }
// STREAM-NEXT:                 }
// STREAM-NEXT:               }
// STREAM-NEXT:             }
// STREAM-NEXT:           }
// STREAM-NEXT:         }
// STREAM-NEXT:       }
// STREAM-NEXT:       taskflow.yield reads(%arg1, %arg3 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>) writes(%arg3 : memref<1x8x8x64xf32>)
// STREAM-NEXT:     }
// STREAM-NEXT:     %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x64x8x8xf32>
// STREAM-NEXT:     %dependency_read_out_10, %dependency_write_out_11 = taskflow.task @Task_4_Task_5_fused dependency_read_in(%dependency_write_out_8 : memref<1x8x8x64xf32>) dependency_write_in(%alloc_9 : memref<1x64x8x8xf32>) value_inputs(%cst_1, %cst_2 : f32, f32) [original_read_memrefs(%alloc_5 : memref<1x8x8x64xf32>), original_write_memrefs(%alloc_9 : memref<1x64x8x8xf32>)] : (memref<1x8x8x64xf32>, memref<1x64x8x8xf32>, f32, f32) -> (memref<1x8x8x64xf32>, memref<1x64x8x8xf32>) {
// STREAM-NEXT:     ^bb0(%arg1: memref<1x8x8x64xf32>, %arg2: memref<1x64x8x8xf32>, %arg3: f32, %arg4: f32):
// STREAM-NEXT:       affine.for %arg5 = 0 to 1 {
// STREAM-NEXT:         affine.for %arg6 = 0 to 64 {
// STREAM-NEXT:           affine.for %arg7 = 0 to 8 {
// STREAM-NEXT:             affine.for %arg8 = 0 to 8 {
// STREAM-NEXT:               %0 = affine.load %arg1[%arg5, %arg7, %arg8, %arg6] : memref<1x8x8x64xf32>
// STREAM-NEXT:               %1 = arith.minimumf %0, %arg3 : f32
// STREAM-NEXT:               %2 = arith.maximumf %1, %arg4 : f32
// STREAM-NEXT:               affine.store %2, %arg2[%arg5, %arg6, %arg7, %arg8] : memref<1x64x8x8xf32>
// STREAM-NEXT:             }
// STREAM-NEXT:           }
// STREAM-NEXT:         }
// STREAM-NEXT:       }
// STREAM-NEXT:       taskflow.yield reads(%arg1 : memref<1x8x8x64xf32>) writes(%arg2 : memref<1x64x8x8xf32>)
// STREAM-NEXT:     }
// STREAM-NEXT:     %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<1x8x8x64xf32>
// STREAM-NEXT:     %dependency_read_out_13, %dependency_write_out_14 = taskflow.task @Task_6 dependency_read_in(%dependency_write_out_11 : memref<1x64x8x8xf32>) dependency_write_in(%alloc_12 : memref<1x8x8x64xf32>) [original_read_memrefs(%alloc_9 : memref<1x64x8x8xf32>), original_write_memrefs(%alloc_12 : memref<1x8x8x64xf32>)] : (memref<1x64x8x8xf32>, memref<1x8x8x64xf32>) -> (memref<1x64x8x8xf32>, memref<1x8x8x64xf32>) {
// STREAM-NEXT:     ^bb0(%arg1: memref<1x64x8x8xf32>, %arg2: memref<1x8x8x64xf32>):
// STREAM-NEXT:       affine.for %arg3 = 0 to 1 {
// STREAM-NEXT:         affine.for %arg4 = 0 to 8 {
// STREAM-NEXT:           affine.for %arg5 = 0 to 8 {
// STREAM-NEXT:             affine.for %arg6 = 0 to 64 {
// STREAM-NEXT:               %0 = affine.load %arg1[%arg3, %arg6, %arg4, %arg5] : memref<1x64x8x8xf32>
// STREAM-NEXT:               affine.store %0, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<1x8x8x64xf32>
// STREAM-NEXT:             }
// STREAM-NEXT:           }
// STREAM-NEXT:         }
// STREAM-NEXT:       }
// STREAM-NEXT:       taskflow.yield reads(%arg1 : memref<1x64x8x8xf32>) writes(%arg2 : memref<1x8x8x64xf32>)
// STREAM-NEXT:     }
// STREAM-NEXT:     %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<1x10x10x64xf32>
// STREAM-NEXT:     %dependency_write_out_16 = taskflow.task @Task_7 dependency_write_in(%alloc_15 : memref<1x10x10x64xf32>) value_inputs(%cst_2 : f32) [original_write_memrefs(%alloc_15 : memref<1x10x10x64xf32>)] : (memref<1x10x10x64xf32>, f32) -> (memref<1x10x10x64xf32>) {
// STREAM-NEXT:     ^bb0(%arg1: memref<1x10x10x64xf32>, %arg2: f32):
// STREAM-NEXT:       affine.for %arg3 = 0 to 1 {
// STREAM-NEXT:         affine.for %arg4 = 0 to 10 {
// STREAM-NEXT:           affine.for %arg5 = 0 to 10 {
// STREAM-NEXT:             affine.for %arg6 = 0 to 64 {
// STREAM-NEXT:               affine.store %arg2, %arg1[%arg3, %arg4, %arg5, %arg6] : memref<1x10x10x64xf32>
// STREAM-NEXT:             }
// STREAM-NEXT:           }
// STREAM-NEXT:         }
// STREAM-NEXT:       }
// STREAM-NEXT:       taskflow.yield writes(%arg1 : memref<1x10x10x64xf32>)
// STREAM-NEXT:     }
// STREAM-NEXT:     %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<1x8x8x64xf32>
// STREAM-NEXT:     %dependency_write_out_18 = taskflow.task @Task_8 dependency_write_in(%alloc_17 : memref<1x8x8x64xf32>) value_inputs(%cst_2 : f32) [original_write_memrefs(%alloc_17 : memref<1x8x8x64xf32>)] : (memref<1x8x8x64xf32>, f32) -> (memref<1x8x8x64xf32>) {
// STREAM-NEXT:     ^bb0(%arg1: memref<1x8x8x64xf32>, %arg2: f32):
// STREAM-NEXT:       affine.for %arg3 = 0 to 1 {
// STREAM-NEXT:         affine.for %arg4 = 0 to 8 {
// STREAM-NEXT:           affine.for %arg5 = 0 to 8 {
// STREAM-NEXT:             affine.for %arg6 = 0 to 64 {
// STREAM-NEXT:               affine.store %arg2, %arg1[%arg3, %arg4, %arg5, %arg6] : memref<1x8x8x64xf32>
// STREAM-NEXT:             }
// STREAM-NEXT:           }
// STREAM-NEXT:         }
// STREAM-NEXT:       }
// STREAM-NEXT:       taskflow.yield writes(%arg1 : memref<1x8x8x64xf32>)
// STREAM-NEXT:     }
// STREAM-NEXT:     %dependency_read_out_19:2, %dependency_write_out_20 = taskflow.task @Task_9 dependency_read_in(%dependency_write_out_16, %dependency_write_out_18 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>) dependency_write_in(%dependency_write_out_18 : memref<1x8x8x64xf32>) value_inputs(%cst : f32) [original_read_memrefs(%alloc_15, %alloc_17 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>), original_write_memrefs(%alloc_17 : memref<1x8x8x64xf32>)] : (memref<1x10x10x64xf32>, memref<1x8x8x64xf32>, memref<1x8x8x64xf32>, f32) -> (memref<1x10x10x64xf32>, memref<1x8x8x64xf32>, memref<1x8x8x64xf32>) {
// STREAM-NEXT:     ^bb0(%arg1: memref<1x10x10x64xf32>, %arg2: memref<1x8x8x64xf32>, %arg3: memref<1x8x8x64xf32>, %arg4: f32):
// STREAM-NEXT:       affine.for %arg5 = 0 to 1 {
// STREAM-NEXT:         affine.for %arg6 = 0 to 8 {
// STREAM-NEXT:           affine.for %arg7 = 0 to 8 {
// STREAM-NEXT:             affine.for %arg8 = 0 to 64 {
// STREAM-NEXT:               affine.for %arg9 = 0 to 3 {
// STREAM-NEXT:                 affine.for %arg10 = 0 to 3 {
// STREAM-NEXT:                   affine.for %arg11 = 0 to 64 {
// STREAM-NEXT:                     %0 = affine.load %arg1[%arg5, %arg6 + %arg9, %arg7 + %arg10, %arg11] : memref<1x10x10x64xf32>
// STREAM-NEXT:                     %1 = affine.load %arg3[%arg5, %arg6, %arg7, %arg8] : memref<1x8x8x64xf32>
// STREAM-NEXT:                     %2 = arith.mulf %0, %arg4 : f32
// STREAM-NEXT:                     %3 = arith.addf %1, %2 : f32
// STREAM-NEXT:                     affine.store %3, %arg3[%arg5, %arg6, %arg7, %arg8] : memref<1x8x8x64xf32>
// STREAM-NEXT:                   }
// STREAM-NEXT:                 }
// STREAM-NEXT:               }
// STREAM-NEXT:             }
// STREAM-NEXT:           }
// STREAM-NEXT:         }
// STREAM-NEXT:       }
// STREAM-NEXT:       taskflow.yield reads(%arg1, %arg3 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>) writes(%arg3 : memref<1x8x8x64xf32>)
// STREAM-NEXT:     }
// STREAM-NEXT:     %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<1x64x8x8xf32>
// STREAM-NEXT:     %dependency_read_out_22:2, %dependency_write_out_23 = taskflow.task @Task_10_Task_11_Task_12_fused_fused dependency_read_in(%dependency_write_out_20, %dependency_read_out : memref<1x8x8x64xf32>, memref<1x64x8x8xf32>) dependency_write_in(%alloc_21 : memref<1x64x8x8xf32>) value_inputs(%cst_1, %cst_2 : f32, f32) [original_read_memrefs(%alloc_17, %arg0 : memref<1x8x8x64xf32>, memref<1x64x8x8xf32>), original_write_memrefs(%alloc_21 : memref<1x64x8x8xf32>)] : (memref<1x8x8x64xf32>, memref<1x64x8x8xf32>, memref<1x64x8x8xf32>, f32, f32) -> (memref<1x8x8x64xf32>, memref<1x64x8x8xf32>, memref<1x64x8x8xf32>) {
// STREAM-NEXT:     ^bb0(%arg1: memref<1x8x8x64xf32>, %arg2: memref<1x64x8x8xf32>, %arg3: memref<1x64x8x8xf32>, %arg4: f32, %arg5: f32):
// STREAM-NEXT:       affine.for %arg6 = 0 to 1 {
// STREAM-NEXT:         affine.for %arg7 = 0 to 64 {
// STREAM-NEXT:           affine.for %arg8 = 0 to 8 {
// STREAM-NEXT:             affine.for %arg9 = 0 to 8 {
// STREAM-NEXT:               %0 = affine.load %arg1[%arg6, %arg8, %arg9, %arg7] : memref<1x8x8x64xf32>
// STREAM-NEXT:               %1 = affine.load %arg2[%arg6, %arg7, %arg8, %arg9] : memref<1x64x8x8xf32>
// STREAM-NEXT:               %2 = arith.addf %0, %1 : f32
// STREAM-NEXT:               %3 = arith.minimumf %2, %arg4 : f32
// STREAM-NEXT:               %4 = arith.maximumf %3, %arg5 : f32
// STREAM-NEXT:               affine.store %4, %arg3[%arg6, %arg7, %arg8, %arg9] : memref<1x64x8x8xf32>
// STREAM-NEXT:             }
// STREAM-NEXT:           }
// STREAM-NEXT:         }
// STREAM-NEXT:       }
// STREAM-NEXT:       taskflow.yield reads(%arg1, %arg2 : memref<1x8x8x64xf32>, memref<1x64x8x8xf32>) writes(%arg3 : memref<1x64x8x8xf32>)
// STREAM-NEXT:     }
// STREAM-NEXT:     return %dependency_write_out_23 : memref<1x64x8x8xf32>
// STREAM-NEXT:   }
// STREAM-NEXT: }


// RESOPT:      module attributes {torch.debug_module_name = "SimpleResNetBlock"} {
// RESOPT-NEXT:   memref.global "private" constant @__constant_64xf32 : memref<64xf32> = dense<0.000000e+00> {alignment = 64 : i64}
// RESOPT-NEXT:   memref.global "private" constant @__constant_64x3x3x64xf32_0 : memref<64x3x3x64xf32> = dense<-0.0151730878> {alignment = 64 : i64}
// RESOPT-NEXT:   memref.global "private" constant @__constant_64x3x3x64xf32 : memref<64x3x3x64xf32> = dense<0.0197670367> {alignment = 64 : i64}
// RESOPT-NEXT:   func.func @forward(%arg0: memref<1x64x8x8xf32>) -> memref<1x64x8x8xf32> {
// RESOPT-NEXT:     %cst = arith.constant 0.0197670367 : f32
// RESOPT-NEXT:     %cst_0 = arith.constant -0.0151730878 : f32
// RESOPT-NEXT:     %cst_1 = arith.constant 3.40282347E+38 : f32
// RESOPT-NEXT:     %cst_2 = arith.constant 0.000000e+00 : f32
// RESOPT-NEXT:     %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x8x8x64xf32>
// RESOPT-NEXT:     %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x10x10x64xf32>
// RESOPT-NEXT:     %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1x8x8x64xf32>
// RESOPT-NEXT:     %dependency_read_out, %dependency_write_out:3 = taskflow.task @Task_1_Task_0_Task_2_utilfused_utilfused dependency_read_in(%arg0 : memref<1x64x8x8xf32>) dependency_write_in(%alloc_3, %alloc, %alloc_4 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>, memref<1x8x8x64xf32>) value_inputs(%cst_2 : f32) [original_read_memrefs(%arg0 : memref<1x64x8x8xf32>), original_write_memrefs(%alloc_3, %alloc, %alloc_4 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>, memref<1x8x8x64xf32>)] {cgra_count = 2 : i32, cgra_shape = "1x2", compiled_ii = 5 : i32, steps = 3 : i32, trip_count = 6400 : i32} : (memref<1x64x8x8xf32>, memref<1x10x10x64xf32>, memref<1x8x8x64xf32>, memref<1x8x8x64xf32>, f32) -> (memref<1x64x8x8xf32>, memref<1x10x10x64xf32>, memref<1x8x8x64xf32>, memref<1x8x8x64xf32>) {
// RESOPT-NEXT:     ^bb0(%arg1: memref<1x64x8x8xf32>, %arg2: memref<1x10x10x64xf32>, %arg3: memref<1x8x8x64xf32>, %arg4: memref<1x8x8x64xf32>, %arg5: f32):
// RESOPT-NEXT:       %c64 = arith.constant 64 : index
// RESOPT-NEXT:       %c10 = arith.constant 10 : index
// RESOPT-NEXT:       %c0 = arith.constant 0 : index
// RESOPT-NEXT:       %c1 = arith.constant 1 : index
// RESOPT-NEXT:       %0 = taskflow.counter from %c0 to %c1 step %c1 attributes {counter_id = 0 : i32, counter_type = "root"} : index
// RESOPT-NEXT:       %1 = taskflow.counter parent(%0 : index) from %c0 to %c10 step %c1 attributes {counter_id = 1 : i32, counter_type = "relay"} : index
// RESOPT-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0 to %c10 step %c1 attributes {counter_id = 2 : i32, counter_type = "relay"} : index
// RESOPT-NEXT:       %3 = taskflow.counter parent(%2 : index) from %c0 to %c64 step %c1 attributes {counter_id = 3 : i32, counter_type = "leaf"} : index
// RESOPT-NEXT:       neura.kernel inputs(%arg5, %arg2, %arg1, %arg3, %arg4 : f32, memref<1x10x10x64xf32>, memref<1x64x8x8xf32>, memref<1x8x8x64xf32>, memref<1x8x8x64xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// RESOPT-NEXT:       ^bb0(%arg6: f32, %arg7: memref<1x10x10x64xf32>, %arg8: memref<1x64x8x8xf32>, %arg9: memref<1x8x8x64xf32>, %arg10: memref<1x8x8x64xf32>):
// RESOPT-NEXT:         %12 = "neura.constant"() <{value = "%input1"}> : () -> !neura.data<memref<1x10x10x64xf32>, i1>
// RESOPT-NEXT:         %13 = neura.counter attributes {counter_id = 0 : i32, counter_type = "root", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 1 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %14 = neura.counter attributes {counter_id = 1 : i32, counter_type = "relay", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 10 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %15 = neura.counter attributes {counter_id = 2 : i32, counter_type = "relay", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 10 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %16 = neura.counter attributes {counter_id = 3 : i32, counter_type = "leaf", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 64 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         neura.store_indexed %12 to %12[%13, %14, %15, %16 : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>] !neura.data<memref<1x10x10x64xf32>, i1> {lhs_value = "%input0"} : !neura.data<memref<1x10x10x64xf32>, i1>
// RESOPT-NEXT:         %17 = neura.counter attributes {counter_id = 0 : i32, counter_type = "root", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 1 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %18 = neura.counter attributes {counter_id = 1 : i32, counter_type = "relay", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 8 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %19 = neura.counter attributes {counter_id = 2 : i32, counter_type = "relay", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 8 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %20 = neura.counter attributes {counter_id = 3 : i32, counter_type = "leaf", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 64 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %21 = neura.load_indexed [%17, %20, %18, %19 : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// RESOPT-NEXT:         neura.store_indexed %21 to [%17, %18, %19, %20 : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>]  {rhs_value = "%input1"} : !neura.data<f32, i1>
// RESOPT-NEXT:         %22 = "neura.constant"() <{value = "%input1"}> : () -> !neura.data<memref<1x8x8x64xf32>, i1>
// RESOPT-NEXT:         %23 = neura.counter attributes {counter_id = 0 : i32, counter_type = "root", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 1 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %24 = neura.counter attributes {counter_id = 1 : i32, counter_type = "relay", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 8 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %25 = neura.counter attributes {counter_id = 2 : i32, counter_type = "relay", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 8 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %26 = neura.counter attributes {counter_id = 3 : i32, counter_type = "leaf", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 64 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         neura.store_indexed %22 to %22[%23, %24, %25, %26 : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>] !neura.data<memref<1x8x8x64xf32>, i1> {lhs_value = "%input0"} : !neura.data<memref<1x8x8x64xf32>, i1>
// RESOPT-NEXT:         neura.yield {yield_type = "void"}
// RESOPT-NEXT:       }
// RESOPT-NEXT:       %c64_20 = arith.constant 64 : index
// RESOPT-NEXT:       %c8 = arith.constant 8 : index
// RESOPT-NEXT:       %c0_21 = arith.constant 0 : index
// RESOPT-NEXT:       %c1_22 = arith.constant 1 : index
// RESOPT-NEXT:       %4 = taskflow.counter from %c0_21 to %c1_22 step %c1_22 attributes {counter_id = 0 : i32, counter_type = "root"} : index
// RESOPT-NEXT:       %5 = taskflow.counter parent(%4 : index) from %c0_21 to %c8 step %c1_22 attributes {counter_id = 1 : i32, counter_type = "relay"} : index
// RESOPT-NEXT:       %6 = taskflow.counter parent(%5 : index) from %c0_21 to %c8 step %c1_22 attributes {counter_id = 2 : i32, counter_type = "relay"} : index
// RESOPT-NEXT:       %7 = taskflow.counter parent(%6 : index) from %c0_21 to %c64_20 step %c1_22 attributes {counter_id = 3 : i32, counter_type = "leaf"} : index
// RESOPT-NEXT:       %c64_23 = arith.constant 64 : index
// RESOPT-NEXT:       %c8_24 = arith.constant 8 : index
// RESOPT-NEXT:       %c0_25 = arith.constant 0 : index
// RESOPT-NEXT:       %c1_26 = arith.constant 1 : index
// RESOPT-NEXT:       %8 = taskflow.counter from %c0_25 to %c1_26 step %c1_26 attributes {counter_id = 0 : i32, counter_type = "root"} : index
// RESOPT-NEXT:       %9 = taskflow.counter parent(%8 : index) from %c0_25 to %c8_24 step %c1_26 attributes {counter_id = 1 : i32, counter_type = "relay"} : index
// RESOPT-NEXT:       %10 = taskflow.counter parent(%9 : index) from %c0_25 to %c8_24 step %c1_26 attributes {counter_id = 2 : i32, counter_type = "relay"} : index
// RESOPT-NEXT:       %11 = taskflow.counter parent(%10 : index) from %c0_25 to %c64_23 step %c1_26 attributes {counter_id = 3 : i32, counter_type = "leaf"} : index
// RESOPT-NEXT:       taskflow.yield reads(%arg1 : memref<1x64x8x8xf32>) writes(%arg2, %arg3, %arg4 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>, memref<1x8x8x64xf32>)
// RESOPT-NEXT:     }
// RESOPT-NEXT:     %dependency_read_out_5:2, %dependency_write_out_6 = taskflow.task @Task_3 dependency_read_in(%dependency_write_out#0, %dependency_write_out#2 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>) dependency_write_in(%dependency_write_out#2 : memref<1x8x8x64xf32>) value_inputs(%cst_0 : f32) [original_read_memrefs(%alloc_3, %alloc_4 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>), original_write_memrefs(%alloc_4 : memref<1x8x8x64xf32>)] {cgra_count = 2 : i32, cgra_shape = "1x2", compiled_ii = 4 : i32, steps = 6 : i32, trip_count = 2359296 : i32} : (memref<1x10x10x64xf32>, memref<1x8x8x64xf32>, memref<1x8x8x64xf32>, f32) -> (memref<1x10x10x64xf32>, memref<1x8x8x64xf32>, memref<1x8x8x64xf32>) {
// RESOPT-NEXT:     ^bb0(%arg1: memref<1x10x10x64xf32>, %arg2: memref<1x8x8x64xf32>, %arg3: memref<1x8x8x64xf32>, %arg4: f32):
// RESOPT-NEXT:       %c3 = arith.constant 3 : index
// RESOPT-NEXT:       %c64 = arith.constant 64 : index
// RESOPT-NEXT:       %c8 = arith.constant 8 : index
// RESOPT-NEXT:       %c0 = arith.constant 0 : index
// RESOPT-NEXT:       %c1 = arith.constant 1 : index
// RESOPT-NEXT:       %0 = taskflow.counter from %c0 to %c1 step %c1 attributes {counter_id = 0 : i32, counter_type = "root"} : index
// RESOPT-NEXT:       %1 = taskflow.counter parent(%0 : index) from %c0 to %c8 step %c1 attributes {counter_id = 1 : i32, counter_type = "relay"} : index
// RESOPT-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0 to %c8 step %c1 attributes {counter_id = 2 : i32, counter_type = "relay"} : index
// RESOPT-NEXT:       %3 = taskflow.counter parent(%2 : index) from %c0 to %c64 step %c1 attributes {counter_id = 3 : i32, counter_type = "relay"} : index
// RESOPT-NEXT:       %4 = taskflow.counter parent(%3 : index) from %c0 to %c3 step %c1 attributes {counter_id = 4 : i32, counter_type = "relay"} : index
// RESOPT-NEXT:       %5 = taskflow.counter parent(%4 : index) from %c0 to %c3 step %c1 attributes {counter_id = 5 : i32, counter_type = "relay"} : index
// RESOPT-NEXT:       %6 = taskflow.counter parent(%5 : index) from %c0 to %c64 step %c1 attributes {counter_id = 6 : i32, counter_type = "leaf"} : index
// RESOPT-NEXT:       neura.kernel inputs(%arg1, %arg3, %arg4 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>, f32) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// RESOPT-NEXT:       ^bb0(%arg5: memref<1x10x10x64xf32>, %arg6: memref<1x8x8x64xf32>, %arg7: f32):
// RESOPT-NEXT:         %7 = neura.counter attributes {counter_id = 0 : i32, counter_type = "root", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 1 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %8 = neura.counter attributes {counter_id = 1 : i32, counter_type = "relay", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 8 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %9 = neura.counter attributes {counter_id = 2 : i32, counter_type = "relay", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 8 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %10 = neura.counter attributes {counter_id = 3 : i32, counter_type = "relay", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 64 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %11 = neura.counter attributes {counter_id = 4 : i32, counter_type = "relay", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 3 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %12 = neura.counter attributes {counter_id = 5 : i32, counter_type = "relay", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 3 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %13 = neura.counter attributes {counter_id = 6 : i32, counter_type = "leaf", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 64 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %14 = "neura.add"(%8, %11) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// RESOPT-NEXT:         %15 = "neura.add"(%9, %12) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// RESOPT-NEXT:         %16 = neura.load_indexed [%7, %14, %15, %13 : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// RESOPT-NEXT:         %17 = neura.load_indexed [%7, %8, %9, %10 : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// RESOPT-NEXT:         %18 = "neura.fmul"(%16) {rhs_value = "%input2"} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// RESOPT-NEXT:         %19 = "neura.fadd"(%17, %18) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// RESOPT-NEXT:         neura.store_indexed %19 to [%7, %8, %9, %10 : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>]  {rhs_value = "%input1"} : !neura.data<f32, i1>
// RESOPT-NEXT:         neura.yield {yield_type = "void"}
// RESOPT-NEXT:       }
// RESOPT-NEXT:       taskflow.yield reads(%arg1, %arg3 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>) writes(%arg3 : memref<1x8x8x64xf32>)
// RESOPT-NEXT:     }
// RESOPT-NEXT:     %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x64x8x8xf32>
// RESOPT-NEXT:     %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x8x8x64xf32>
// RESOPT-NEXT:     %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x10x10x64xf32>
// RESOPT-NEXT:     %dependency_read_out_10, %dependency_write_out_11:2 = taskflow.task @Task_4_Task_5_fused_Task_7_utilfused dependency_read_in(%dependency_write_out_6 : memref<1x8x8x64xf32>) dependency_write_in(%alloc_7, %alloc_9 : memref<1x64x8x8xf32>, memref<1x10x10x64xf32>) value_inputs(%cst_1, %cst_2 : f32, f32) [original_read_memrefs(%alloc_4 : memref<1x8x8x64xf32>), original_write_memrefs(%alloc_7, %alloc_9 : memref<1x64x8x8xf32>, memref<1x10x10x64xf32>)] {cgra_count = 2 : i32, cgra_shape = "1x2", compiled_ii = 4 : i32, steps = 7 : i32, trip_count = 6400 : i32} : (memref<1x8x8x64xf32>, memref<1x64x8x8xf32>, memref<1x10x10x64xf32>, f32, f32) -> (memref<1x8x8x64xf32>, memref<1x64x8x8xf32>, memref<1x10x10x64xf32>) {
// RESOPT-NEXT:     ^bb0(%arg1: memref<1x8x8x64xf32>, %arg2: memref<1x64x8x8xf32>, %arg3: memref<1x10x10x64xf32>, %arg4: f32, %arg5: f32):
// RESOPT-NEXT:       %c8 = arith.constant 8 : index
// RESOPT-NEXT:       %c64 = arith.constant 64 : index
// RESOPT-NEXT:       %c0 = arith.constant 0 : index
// RESOPT-NEXT:       %c1 = arith.constant 1 : index
// RESOPT-NEXT:       %0 = taskflow.counter from %c0 to %c1 step %c1 attributes {counter_id = 0 : i32, counter_type = "root"} : index
// RESOPT-NEXT:       %1 = taskflow.counter parent(%0 : index) from %c0 to %c64 step %c1 attributes {counter_id = 1 : i32, counter_type = "relay"} : index
// RESOPT-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0 to %c8 step %c1 attributes {counter_id = 2 : i32, counter_type = "relay"} : index
// RESOPT-NEXT:       %3 = taskflow.counter parent(%2 : index) from %c0 to %c8 step %c1 attributes {counter_id = 3 : i32, counter_type = "leaf"} : index
// RESOPT-NEXT:       neura.kernel inputs(%arg1, %arg4, %arg5, %arg2, %arg3 : memref<1x8x8x64xf32>, f32, f32, memref<1x64x8x8xf32>, memref<1x10x10x64xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// RESOPT-NEXT:       ^bb0(%arg6: memref<1x8x8x64xf32>, %arg7: f32, %arg8: f32, %arg9: memref<1x64x8x8xf32>, %arg10: memref<1x10x10x64xf32>):
// RESOPT-NEXT:         %8 = "neura.constant"() <{value = "%input1"}> : () -> !neura.data<f32, i1>
// RESOPT-NEXT:         %9 = "neura.constant"() <{value = "%input2"}> : () -> !neura.data<f32, i1>
// RESOPT-NEXT:         %10 = neura.counter attributes {counter_id = 0 : i32, counter_type = "root", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 1 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %11 = neura.counter attributes {counter_id = 1 : i32, counter_type = "relay", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 64 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %12 = neura.counter attributes {counter_id = 2 : i32, counter_type = "relay", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 8 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %13 = neura.counter attributes {counter_id = 3 : i32, counter_type = "leaf", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 8 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %14 = neura.load_indexed [%10, %12, %13, %11 : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// RESOPT-NEXT:         %15 = "neura.fcmp"(%14, %8) <{cmpType = "olt"}> : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// RESOPT-NEXT:         %16 = "neura.sel"(%15, %14, %8) : (!neura.data<f32, i1>, !neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// RESOPT-NEXT:         %17 = "neura.fcmp"(%16, %9) <{cmpType = "ogt"}> : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// RESOPT-NEXT:         %18 = "neura.sel"(%17, %16, %9) : (!neura.data<f32, i1>, !neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// RESOPT-NEXT:         neura.store_indexed %18 to [%10, %11, %12, %13 : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>]  {rhs_value = "%input3"} : !neura.data<f32, i1>
// RESOPT-NEXT:         %19 = "neura.constant"() <{value = "%input1"}> : () -> !neura.data<memref<1x10x10x64xf32>, i1>
// RESOPT-NEXT:         %20 = neura.counter attributes {counter_id = 0 : i32, counter_type = "root", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 1 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %21 = neura.counter attributes {counter_id = 1 : i32, counter_type = "relay", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 10 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %22 = neura.counter attributes {counter_id = 2 : i32, counter_type = "relay", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 10 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %23 = neura.counter attributes {counter_id = 3 : i32, counter_type = "leaf", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 64 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         neura.store_indexed %19 to %19[%20, %21, %22, %23 : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>] !neura.data<memref<1x10x10x64xf32>, i1> {lhs_value = "%input0"} : !neura.data<memref<1x10x10x64xf32>, i1>
// RESOPT-NEXT:         neura.yield {yield_type = "void"}
// RESOPT-NEXT:       }
// RESOPT-NEXT:       %c64_20 = arith.constant 64 : index
// RESOPT-NEXT:       %c10 = arith.constant 10 : index
// RESOPT-NEXT:       %c0_21 = arith.constant 0 : index
// RESOPT-NEXT:       %c1_22 = arith.constant 1 : index
// RESOPT-NEXT:       %4 = taskflow.counter from %c0_21 to %c1_22 step %c1_22 attributes {counter_id = 0 : i32, counter_type = "root"} : index
// RESOPT-NEXT:       %5 = taskflow.counter parent(%4 : index) from %c0_21 to %c10 step %c1_22 attributes {counter_id = 1 : i32, counter_type = "relay"} : index
// RESOPT-NEXT:       %6 = taskflow.counter parent(%5 : index) from %c0_21 to %c10 step %c1_22 attributes {counter_id = 2 : i32, counter_type = "relay"} : index
// RESOPT-NEXT:       %7 = taskflow.counter parent(%6 : index) from %c0_21 to %c64_20 step %c1_22 attributes {counter_id = 3 : i32, counter_type = "leaf"} : index
// RESOPT-NEXT:       taskflow.yield reads(%arg1 : memref<1x8x8x64xf32>) writes(%arg2, %arg3 : memref<1x64x8x8xf32>, memref<1x10x10x64xf32>)
// RESOPT-NEXT:     }
// RESOPT-NEXT:     %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<1x8x8x64xf32>
// RESOPT-NEXT:     %dependency_read_out_13, %dependency_write_out_14:2 = taskflow.task @Task_6_Task_8_utilfused dependency_read_in(%dependency_write_out_11#0 : memref<1x64x8x8xf32>) dependency_write_in(%alloc_8, %alloc_12 : memref<1x8x8x64xf32>, memref<1x8x8x64xf32>) value_inputs(%cst_2 : f32) [original_read_memrefs(%alloc_7 : memref<1x64x8x8xf32>), original_write_memrefs(%alloc_8, %alloc_12 : memref<1x8x8x64xf32>, memref<1x8x8x64xf32>)] {cgra_count = 2 : i32, cgra_shape = "1x2", compiled_ii = 3 : i32, steps = 3 : i32, trip_count = 4096 : i32} : (memref<1x64x8x8xf32>, memref<1x8x8x64xf32>, memref<1x8x8x64xf32>, f32) -> (memref<1x64x8x8xf32>, memref<1x8x8x64xf32>, memref<1x8x8x64xf32>) {
// RESOPT-NEXT:     ^bb0(%arg1: memref<1x64x8x8xf32>, %arg2: memref<1x8x8x64xf32>, %arg3: memref<1x8x8x64xf32>, %arg4: f32):
// RESOPT-NEXT:       %c64 = arith.constant 64 : index
// RESOPT-NEXT:       %c8 = arith.constant 8 : index
// RESOPT-NEXT:       %c0 = arith.constant 0 : index
// RESOPT-NEXT:       %c1 = arith.constant 1 : index
// RESOPT-NEXT:       %0 = taskflow.counter from %c0 to %c1 step %c1 attributes {counter_id = 0 : i32, counter_type = "root"} : index
// RESOPT-NEXT:       %1 = taskflow.counter parent(%0 : index) from %c0 to %c8 step %c1 attributes {counter_id = 1 : i32, counter_type = "relay"} : index
// RESOPT-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0 to %c8 step %c1 attributes {counter_id = 2 : i32, counter_type = "relay"} : index
// RESOPT-NEXT:       %3 = taskflow.counter parent(%2 : index) from %c0 to %c64 step %c1 attributes {counter_id = 3 : i32, counter_type = "leaf"} : index
// RESOPT-NEXT:       neura.kernel inputs(%arg1, %arg2, %arg4, %arg3 : memref<1x64x8x8xf32>, memref<1x8x8x64xf32>, f32, memref<1x8x8x64xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// RESOPT-NEXT:       ^bb0(%arg5: memref<1x64x8x8xf32>, %arg6: memref<1x8x8x64xf32>, %arg7: f32, %arg8: memref<1x8x8x64xf32>):
// RESOPT-NEXT:         %8 = neura.counter attributes {counter_id = 0 : i32, counter_type = "root", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 1 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %9 = neura.counter attributes {counter_id = 1 : i32, counter_type = "relay", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 8 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %10 = neura.counter attributes {counter_id = 2 : i32, counter_type = "relay", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 8 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %11 = neura.counter attributes {counter_id = 3 : i32, counter_type = "leaf", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 64 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %12 = neura.load_indexed [%8, %11, %9, %10 : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// RESOPT-NEXT:         neura.store_indexed %12 to [%8, %9, %10, %11 : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>]  {rhs_value = "%input1"} : !neura.data<f32, i1>
// RESOPT-NEXT:         %13 = "neura.constant"() <{value = "%input1"}> : () -> !neura.data<memref<1x8x8x64xf32>, i1>
// RESOPT-NEXT:         %14 = neura.counter attributes {counter_id = 0 : i32, counter_type = "root", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 1 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %15 = neura.counter attributes {counter_id = 1 : i32, counter_type = "relay", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 8 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %16 = neura.counter attributes {counter_id = 2 : i32, counter_type = "relay", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 8 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %17 = neura.counter attributes {counter_id = 3 : i32, counter_type = "leaf", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 64 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         neura.store_indexed %13 to %13[%14, %15, %16, %17 : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>] !neura.data<memref<1x8x8x64xf32>, i1> {lhs_value = "%input0"} : !neura.data<memref<1x8x8x64xf32>, i1>
// RESOPT-NEXT:         neura.yield {yield_type = "void"}
// RESOPT-NEXT:       }
// RESOPT-NEXT:       %c64_20 = arith.constant 64 : index
// RESOPT-NEXT:       %c8_21 = arith.constant 8 : index
// RESOPT-NEXT:       %c0_22 = arith.constant 0 : index
// RESOPT-NEXT:       %c1_23 = arith.constant 1 : index
// RESOPT-NEXT:       %4 = taskflow.counter from %c0_22 to %c1_23 step %c1_23 attributes {counter_id = 0 : i32, counter_type = "root"} : index
// RESOPT-NEXT:       %5 = taskflow.counter parent(%4 : index) from %c0_22 to %c8_21 step %c1_23 attributes {counter_id = 1 : i32, counter_type = "relay"} : index
// RESOPT-NEXT:       %6 = taskflow.counter parent(%5 : index) from %c0_22 to %c8_21 step %c1_23 attributes {counter_id = 2 : i32, counter_type = "relay"} : index
// RESOPT-NEXT:       %7 = taskflow.counter parent(%6 : index) from %c0_22 to %c64_20 step %c1_23 attributes {counter_id = 3 : i32, counter_type = "leaf"} : index
// RESOPT-NEXT:       taskflow.yield reads(%arg1 : memref<1x64x8x8xf32>) writes(%arg2, %arg3 : memref<1x8x8x64xf32>, memref<1x8x8x64xf32>)
// RESOPT-NEXT:     }
// RESOPT-NEXT:     %dependency_read_out_15:2, %dependency_write_out_16 = taskflow.task @Task_9 dependency_read_in(%dependency_write_out_11#1, %dependency_write_out_14#1 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>) dependency_write_in(%dependency_write_out_14#1 : memref<1x8x8x64xf32>) value_inputs(%cst : f32) [original_read_memrefs(%alloc_9, %alloc_12 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>), original_write_memrefs(%alloc_12 : memref<1x8x8x64xf32>)] {cgra_count = 2 : i32, cgra_shape = "1x2", compiled_ii = 4 : i32, steps = 6 : i32, trip_count = 2359296 : i32} : (memref<1x10x10x64xf32>, memref<1x8x8x64xf32>, memref<1x8x8x64xf32>, f32) -> (memref<1x10x10x64xf32>, memref<1x8x8x64xf32>, memref<1x8x8x64xf32>) {
// RESOPT-NEXT:     ^bb0(%arg1: memref<1x10x10x64xf32>, %arg2: memref<1x8x8x64xf32>, %arg3: memref<1x8x8x64xf32>, %arg4: f32):
// RESOPT-NEXT:       %c3 = arith.constant 3 : index
// RESOPT-NEXT:       %c64 = arith.constant 64 : index
// RESOPT-NEXT:       %c8 = arith.constant 8 : index
// RESOPT-NEXT:       %c0 = arith.constant 0 : index
// RESOPT-NEXT:       %c1 = arith.constant 1 : index
// RESOPT-NEXT:       %0 = taskflow.counter from %c0 to %c1 step %c1 attributes {counter_id = 0 : i32, counter_type = "root"} : index
// RESOPT-NEXT:       %1 = taskflow.counter parent(%0 : index) from %c0 to %c8 step %c1 attributes {counter_id = 1 : i32, counter_type = "relay"} : index
// RESOPT-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0 to %c8 step %c1 attributes {counter_id = 2 : i32, counter_type = "relay"} : index
// RESOPT-NEXT:       %3 = taskflow.counter parent(%2 : index) from %c0 to %c64 step %c1 attributes {counter_id = 3 : i32, counter_type = "relay"} : index
// RESOPT-NEXT:       %4 = taskflow.counter parent(%3 : index) from %c0 to %c3 step %c1 attributes {counter_id = 4 : i32, counter_type = "relay"} : index
// RESOPT-NEXT:       %5 = taskflow.counter parent(%4 : index) from %c0 to %c3 step %c1 attributes {counter_id = 5 : i32, counter_type = "relay"} : index
// RESOPT-NEXT:       %6 = taskflow.counter parent(%5 : index) from %c0 to %c64 step %c1 attributes {counter_id = 6 : i32, counter_type = "leaf"} : index
// RESOPT-NEXT:       neura.kernel inputs(%arg1, %arg3, %arg4 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>, f32) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// RESOPT-NEXT:       ^bb0(%arg5: memref<1x10x10x64xf32>, %arg6: memref<1x8x8x64xf32>, %arg7: f32):
// RESOPT-NEXT:         %7 = neura.counter attributes {counter_id = 0 : i32, counter_type = "root", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 1 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %8 = neura.counter attributes {counter_id = 1 : i32, counter_type = "relay", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 8 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %9 = neura.counter attributes {counter_id = 2 : i32, counter_type = "relay", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 8 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %10 = neura.counter attributes {counter_id = 3 : i32, counter_type = "relay", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 64 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %11 = neura.counter attributes {counter_id = 4 : i32, counter_type = "relay", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 3 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %12 = neura.counter attributes {counter_id = 5 : i32, counter_type = "relay", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 3 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %13 = neura.counter attributes {counter_id = 6 : i32, counter_type = "leaf", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 64 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %14 = "neura.add"(%8, %11) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// RESOPT-NEXT:         %15 = "neura.add"(%9, %12) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// RESOPT-NEXT:         %16 = neura.load_indexed [%7, %14, %15, %13 : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// RESOPT-NEXT:         %17 = neura.load_indexed [%7, %8, %9, %10 : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// RESOPT-NEXT:         %18 = "neura.fmul"(%16) {rhs_value = "%input2"} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// RESOPT-NEXT:         %19 = "neura.fadd"(%17, %18) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// RESOPT-NEXT:         neura.store_indexed %19 to [%7, %8, %9, %10 : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>]  {rhs_value = "%input1"} : !neura.data<f32, i1>
// RESOPT-NEXT:         neura.yield {yield_type = "void"}
// RESOPT-NEXT:       }
// RESOPT-NEXT:       taskflow.yield reads(%arg1, %arg3 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>) writes(%arg3 : memref<1x8x8x64xf32>)
// RESOPT-NEXT:     }
// RESOPT-NEXT:     %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<1x64x8x8xf32>
// RESOPT-NEXT:     %dependency_read_out_18:2, %dependency_write_out_19 = taskflow.task @Task_10_Task_11_Task_12_fused_fused dependency_read_in(%dependency_write_out_16, %dependency_read_out : memref<1x8x8x64xf32>, memref<1x64x8x8xf32>) dependency_write_in(%alloc_17 : memref<1x64x8x8xf32>) value_inputs(%cst_1, %cst_2 : f32, f32) [original_read_memrefs(%alloc_12, %arg0 : memref<1x8x8x64xf32>, memref<1x64x8x8xf32>), original_write_memrefs(%alloc_17 : memref<1x64x8x8xf32>)] {cgra_count = 2 : i32, cgra_shape = "1x2", compiled_ii = 7 : i32, steps = 8 : i32, trip_count = 4096 : i32} : (memref<1x8x8x64xf32>, memref<1x64x8x8xf32>, memref<1x64x8x8xf32>, f32, f32) -> (memref<1x8x8x64xf32>, memref<1x64x8x8xf32>, memref<1x64x8x8xf32>) {
// RESOPT-NEXT:     ^bb0(%arg1: memref<1x8x8x64xf32>, %arg2: memref<1x64x8x8xf32>, %arg3: memref<1x64x8x8xf32>, %arg4: f32, %arg5: f32):
// RESOPT-NEXT:       %c8 = arith.constant 8 : index
// RESOPT-NEXT:       %c64 = arith.constant 64 : index
// RESOPT-NEXT:       %c0 = arith.constant 0 : index
// RESOPT-NEXT:       %c1 = arith.constant 1 : index
// RESOPT-NEXT:       %0 = taskflow.counter from %c0 to %c1 step %c1 attributes {counter_id = 0 : i32, counter_type = "root"} : index
// RESOPT-NEXT:       %1 = taskflow.counter parent(%0 : index) from %c0 to %c64 step %c1 attributes {counter_id = 1 : i32, counter_type = "relay"} : index
// RESOPT-NEXT:       %2 = taskflow.counter parent(%1 : index) from %c0 to %c8 step %c1 attributes {counter_id = 2 : i32, counter_type = "relay"} : index
// RESOPT-NEXT:       %3 = taskflow.counter parent(%2 : index) from %c0 to %c8 step %c1 attributes {counter_id = 3 : i32, counter_type = "leaf"} : index
// RESOPT-NEXT:       neura.kernel inputs(%arg1, %arg2, %arg4, %arg5, %arg3 : memref<1x8x8x64xf32>, memref<1x64x8x8xf32>, f32, f32, memref<1x64x8x8xf32>) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// RESOPT-NEXT:       ^bb0(%arg6: memref<1x8x8x64xf32>, %arg7: memref<1x64x8x8xf32>, %arg8: f32, %arg9: f32, %arg10: memref<1x64x8x8xf32>):
// RESOPT-NEXT:         %4 = "neura.constant"() <{value = "%input2"}> : () -> !neura.data<f32, i1>
// RESOPT-NEXT:         %5 = "neura.constant"() <{value = "%input3"}> : () -> !neura.data<f32, i1>
// RESOPT-NEXT:         %6 = neura.counter attributes {counter_id = 0 : i32, counter_type = "root", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 1 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %7 = neura.counter attributes {counter_id = 1 : i32, counter_type = "relay", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 64 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %8 = neura.counter attributes {counter_id = 2 : i32, counter_type = "relay", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 8 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %9 = neura.counter attributes {counter_id = 3 : i32, counter_type = "leaf", lower_bound_value = 0 : index, step_value = 1 : index, upper_bound_value = 8 : index} -> !neura.data<index, i1>
// RESOPT-NEXT:         %10 = neura.load_indexed [%6, %8, %9, %7 : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>]  {lhs_value = "%input0"} : !neura.data<f32, i1>
// RESOPT-NEXT:         %11 = neura.load_indexed [%6, %7, %8, %9 : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>]  {lhs_value = "%input1"} : !neura.data<f32, i1>
// RESOPT-NEXT:         %12 = "neura.fadd"(%10, %11) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// RESOPT-NEXT:         %13 = "neura.fcmp"(%12, %4) <{cmpType = "olt"}> : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// RESOPT-NEXT:         %14 = "neura.sel"(%13, %12, %4) : (!neura.data<f32, i1>, !neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// RESOPT-NEXT:         %15 = "neura.fcmp"(%14, %5) <{cmpType = "ogt"}> : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// RESOPT-NEXT:         %16 = "neura.sel"(%15, %14, %5) : (!neura.data<f32, i1>, !neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// RESOPT-NEXT:         neura.store_indexed %16 to [%6, %7, %8, %9 : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>]  {rhs_value = "%input4"} : !neura.data<f32, i1>
// RESOPT-NEXT:         neura.yield {yield_type = "void"}
// RESOPT-NEXT:       }
// RESOPT-NEXT:       taskflow.yield reads(%arg1, %arg2 : memref<1x8x8x64xf32>, memref<1x64x8x8xf32>) writes(%arg3 : memref<1x64x8x8xf32>)
// RESOPT-NEXT:     }
// RESOPT-NEXT:     return %dependency_write_out_19 : memref<1x64x8x8xf32>
// RESOPT-NEXT:   }
// RESOPT-NEXT: }



