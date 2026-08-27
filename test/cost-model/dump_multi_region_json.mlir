// RUN: mlir-neura-opt %s \
// RUN:   --architecture-spec=%S/../arch_spec/architecture_4x4.yaml \
// RUN:   --dump-dfg-json="output-file=%t.json"
// RUN: python3 -c 'import json; d=json.load(open("%t.json")); assert isinstance(d, list) and len(d) == 2 and {x["name"] for x in d} == {"first", "second"}'
// RUN: FileCheck %s --input-file=%t.json

// Multiple accelerator regions must be emitted as one valid JSON document,
// with a stable name on each element. A stream of two bare objects is invalid.

module {
  func.func @first() attributes {accelerator = "neura"} {
    %c = "neura.grant_once"() <{constant_value = 0 : i64}>
        : () -> !neura.data<i64, i1>
    neura.return_void %c : !neura.data<i64, i1>
    neura.yield
  }
  func.func @second() attributes {accelerator = "neura"} {
    %c = "neura.grant_once"() <{constant_value = 1 : i64}>
        : () -> !neura.data<i64, i1>
    neura.return_void %c : !neura.data<i64, i1>
    neura.yield
  }
}

// CHECK: [
// CHECK: "name": "first"
// CHECK: "name": "second"
