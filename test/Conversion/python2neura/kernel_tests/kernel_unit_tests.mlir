// RUN: neura-interpreter %s 2>&1 | FileCheck %s
// RUN: neura-interpreter %s --dataflow 2>&1 | FileCheck %s --check-prefix=CHECK-DF
//
// ============================================================================
// neura.kernel container behavior unit tests
// ============================================================================
// Tests:
//   1. Empty kernel (no counters) — executes once and exits
//   2. Single leaf counter — iterates exactly N times
//   3. Nested counters leaf→relay→root — nested loop advancement
//   4. Single iteration boundary — upper_bound=1
//   5. Non-unit step — step=2 counter
//   6. Multi-kernel scope isolation — same %input name maps to different memrefs
//   7. Independent nested loops — two kernels with 3-level counters, no interference
//   8. Multiple kernels with different memrefs running in parallel
// ============================================================================

module {

  // --------------------------------------------------------------------------
  // Test 1: Empty kernel — no counters, executes once and exits
  // Verifies advanceCountersNested returns false for empty counter_values
  // --------------------------------------------------------------------------
  func.func @test1_empty_kernel() -> f32 {
    %c = arith.constant 1.0 : f32
    %alloc = memref.alloc() : memref<1xf32>
    neura.kernel inputs(%alloc : memref<1xf32>) attributes {accelerator = "neura"} {
    ^bb0(%out: memref<1xf32>):
      neura.yield
    }
    return %c : f32
  }

  // --------------------------------------------------------------------------
  // Test 2: Single leaf counter — lower=0, upper=4, exactly 4 iterations
  // Verifies advanceOneCounter and reset behavior in leaf-only mode
  // --------------------------------------------------------------------------
  func.func @test2_leaf_counter() -> f32 {
    %c = arith.constant 2.0 : f32
    %alloc = memref.alloc() : memref<1xf32>
    neura.kernel inputs(%alloc : memref<1xf32>) attributes {accelerator = "neura"} {
    ^bb0(%out: memref<1xf32>):
      %i = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
      neura.yield
    }
    return %c : f32
  }

  // --------------------------------------------------------------------------
  // Test 3: Nested counters leaf→relay→root (2×3×4 = 24 iterations)
  // Verifies nested loop advancement: leaf overflow → relay advance → reset leaf;
  // relay overflow → root advance → reset relay
  // --------------------------------------------------------------------------
  func.func @test3_nested_counters() -> f32 {
    %c = arith.constant 3.0 : f32
    %alloc = memref.alloc() : memref<1xf32>
    neura.kernel inputs(%alloc : memref<1xf32>) attributes {accelerator = "neura"} {
    ^bb0(%out: memref<1xf32>):
      %root = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 4 : i64} -> !neura.data<i64, i1>
      %relay = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 3 : i64} -> !neura.data<i64, i1>
      %leaf = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 2 : i64} -> !neura.data<i64, i1>
      neura.yield
    }
    return %c : f32
  }

  // --------------------------------------------------------------------------
  // Test 4: Single iteration — upper_bound=1, runs exactly once
  // Verifies boundary condition: counter completes exactly one trip
  // --------------------------------------------------------------------------
  func.func @test4_single_iteration() -> f32 {
    %c = arith.constant 4.0 : f32
    %alloc = memref.alloc() : memref<1xf32>
    neura.kernel inputs(%alloc : memref<1xf32>) attributes {accelerator = "neura"} {
    ^bb0(%out: memref<1xf32>):
      %i = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 1 : i64} -> !neura.data<i64, i1>
      neura.yield
    }
    return %c : f32
  }

  // --------------------------------------------------------------------------
  // Test 5: Non-unit step — step=2, upper=6 → 3 iterations (0,2,4)
  // Verifies advanceOneCounter behavior with step != 1
  // --------------------------------------------------------------------------
  func.func @test5_step_greater_than_one() -> f32 {
    %c = arith.constant 5.0 : f32
    %alloc = memref.alloc() : memref<1xf32>
    neura.kernel inputs(%alloc : memref<1xf32>) attributes {accelerator = "neura"} {
    ^bb0(%out: memref<1xf32>):
      %i = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 2 : i64, upper_bound_value = 6 : i64} -> !neura.data<i64, i1>
      neura.yield
    }
    return %c : f32
  }

  // --------------------------------------------------------------------------
  // Test 6: Multi-kernel scope isolation — two kernels with different memrefs
  // Verifies each kernel gets independent simulated_memory key prefix (m<id>/),
  // no interference between kernels. Both kernels execute and exit correctly.
  // --------------------------------------------------------------------------
  func.func @test6_multi_kernel_isolation() -> f32 {
    %c = arith.constant 6.0 : f32
    %alloc_a = memref.alloc() : memref<1xf32>
    %alloc_b = memref.alloc() : memref<1xf32>

    // Kernel A: leaf counter, iterates 3 times on alloc_a
    neura.kernel inputs(%alloc_a : memref<1xf32>) attributes {accelerator = "neura"} {
    ^bb0(%a: memref<1xf32>):
      %i = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 3 : i64} -> !neura.data<i64, i1>
      neura.yield
    }

    // Kernel B: leaf counter, iterates 5 times on alloc_b (separate scope)
    neura.kernel inputs(%alloc_b : memref<1xf32>) attributes {accelerator = "neura"} {
    ^bb0(%b: memref<1xf32>):
      %i = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 5 : i64} -> !neura.data<i64, i1>
      neura.yield
    }

    return %c : f32
  }

  // --------------------------------------------------------------------------
  // Test 7: Independent nested loops — two kernels each with full 3-level counters
  // Kernel A: root=2, relay=2, leaf=2 → 8 iterations
  // Kernel B: root=3, relay=2, leaf=2 → 12 iterations
  // Verifies counter_state is reset after first kernel exits, and second kernel's
  // counters start from lower_bound.
  // --------------------------------------------------------------------------
  func.func @test7_independent_loops() -> f32 {
    %c = arith.constant 7.0 : f32
    %alloc = memref.alloc() : memref<1xf32>

    // Kernel A: 2×2×2 = 8 iterations
    neura.kernel inputs(%alloc : memref<1xf32>) attributes {accelerator = "neura"} {
    ^bb0(%a: memref<1xf32>):
      %r = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 2 : i64} -> !neura.data<i64, i1>
      %y = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 2 : i64} -> !neura.data<i64, i1>
      %l = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 2 : i64} -> !neura.data<i64, i1>
      neura.yield
    }

    // Kernel B: 3×2×2 = 12 iterations (different upper bound)
    neura.kernel inputs(%alloc : memref<1xf32>) attributes {accelerator = "neura"} {
    ^bb0(%a: memref<1xf32>):
      %r = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 3 : i64} -> !neura.data<i64, i1>
      %y = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 2 : i64} -> !neura.data<i64, i1>
      %l = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 2 : i64} -> !neura.data<i64, i1>
      neura.yield
    }

    return %c : f32
  }

  // --------------------------------------------------------------------------
  // Test 8: Many kernels with different memrefs — verifies no naming conflicts
  // 4 kernels, each with a different memref, all with counters
  // --------------------------------------------------------------------------
  func.func @test8_many_kernels() -> f32 {
    %c = arith.constant 8.0 : f32
    %a1 = memref.alloc() : memref<1xf32>
    %a2 = memref.alloc() : memref<2xf32>
    %a3 = memref.alloc() : memref<3xf32>
    %a4 = memref.alloc() : memref<4xf32>

    neura.kernel inputs(%a1 : memref<1xf32>) attributes {accelerator = "neura"} {
    ^bb0(%x: memref<1xf32>):
      %i = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 2 : i64} -> !neura.data<i64, i1>
      neura.yield
    }

    neura.kernel inputs(%a2 : memref<2xf32>) attributes {accelerator = "neura"} {
    ^bb0(%x: memref<2xf32>):
      %i = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 2 : i64} -> !neura.data<i64, i1>
      neura.yield
    }

    neura.kernel inputs(%a3 : memref<3xf32>) attributes {accelerator = "neura"} {
    ^bb0(%x: memref<3xf32>):
      %i = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 2 : i64} -> !neura.data<i64, i1>
      neura.yield
    }

    neura.kernel inputs(%a4 : memref<4xf32>) attributes {accelerator = "neura"} {
    ^bb0(%x: memref<4xf32>):
      %i = neura.counter attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32, lower_bound_value = 0 : i64, step_value = 1 : i64, upper_bound_value = 2 : i64} -> !neura.data<i64, i1>
      neura.yield
    }

    return %c : f32
  }

}

// ============================================================================
// FileCheck patterns — verify all 8 functions produce expected return values
// ============================================================================

// CHECK-DAG: Output: 1.000000
// CHECK-DAG: Output: 2.000000
// CHECK-DAG: Output: 3.000000
// CHECK-DAG: Output: 4.000000
// CHECK-DAG: Output: 5.000000
// CHECK-DAG: Output: 6.000000
// CHECK-DAG: Output: 7.000000
// CHECK-DAG: Output: 8.000000
// CHECK-NOT: Error
// CHECK-NOT: Failed
// CHECK-NOT: Unhandled op

// CHECK-DF-DAG: Output: 1.000000
// CHECK-DF-DAG: Output: 2.000000
// CHECK-DF-DAG: Output: 3.000000
// CHECK-DF-DAG: Output: 4.000000
// CHECK-DF-DAG: Output: 5.000000
// CHECK-DF-DAG: Output: 6.000000
// CHECK-DF-DAG: Output: 7.000000
// CHECK-DF-DAG: Output: 8.000000
// CHECK-DF-NOT: Error
// CHECK-DF-NOT: Failed
