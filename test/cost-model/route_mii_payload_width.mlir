// RUN: mlir-neura-opt %s --cost-model-analytical \
// RUN:   --architecture-spec=%S/../arch_spec/architecture_4x4.yaml 2>&1 | FileCheck %s

// 25 i64 data moves require two 32-bit channels each. The default 4x4 mesh has
// 48 directed links, so the correct channel demand of 50 gives RouteMII 2;
// treating !neura.data<i64, i1> as opaque would incorrectly report 25 and 1.

module {
  func.func @route_mii_uses_payload_width() attributes {accelerator = "neura"} {
    %initial = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
    %move0 = "neura.data_mov"(%initial) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %move1 = "neura.data_mov"(%move0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %move2 = "neura.data_mov"(%move1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %move3 = "neura.data_mov"(%move2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %move4 = "neura.data_mov"(%move3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %move5 = "neura.data_mov"(%move4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %move6 = "neura.data_mov"(%move5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %move7 = "neura.data_mov"(%move6) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %move8 = "neura.data_mov"(%move7) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %move9 = "neura.data_mov"(%move8) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %move10 = "neura.data_mov"(%move9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %move11 = "neura.data_mov"(%move10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %move12 = "neura.data_mov"(%move11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %move13 = "neura.data_mov"(%move12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %move14 = "neura.data_mov"(%move13) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %move15 = "neura.data_mov"(%move14) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %move16 = "neura.data_mov"(%move15) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %move17 = "neura.data_mov"(%move16) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %move18 = "neura.data_mov"(%move17) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %move19 = "neura.data_mov"(%move18) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %move20 = "neura.data_mov"(%move19) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %move21 = "neura.data_mov"(%move20) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %move22 = "neura.data_mov"(%move21) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %move23 = "neura.data_mov"(%move22) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %move24 = "neura.data_mov"(%move23) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    neura.return_void %move24 : !neura.data<i64, i1>
    neura.yield
  }
}

// CHECK: route_mii=2 (moves=25 channel_demand=50 links=48 link_bw=32)
