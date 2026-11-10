// A per-CGRA vector accumulation loop whose exit value is combined across
// CGRAs by a global reduce.
// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-input-arg-to-const \
// RUN:   --fold-constant \
// RUN:   --canonicalize-return \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   --insert-data-mov \
// RUN:   --assign-reduce-id \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic" \
// RUN:   --architecture-spec=%S/../../arch_spec/architecture_with_global_reduce.yaml \
// RUN:   --generate-code -o %t-mapping.mlir
// RUN: FileCheck %s --input-file=%t-mapping.mlir -check-prefix=MAPPING
// RUN: FileCheck %s --input-file=tmp-generated-instructions.yaml --check-prefix=YAML

func.func @kernel(%a: !llvm.ptr) -> i32 {
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %c4 = llvm.mlir.constant(4 : i64) : i64
  %n = llvm.mlir.constant(32 : i64) : i64
  %zero = llvm.mlir.constant(dense<0> : vector<4xi32>) : vector<4xi32>
  llvm.br ^bb1(%c0, %zero : i64, vector<4xi32>)

^bb1(%i: i64, %acc: vector<4xi32>):
  %p = llvm.getelementptr inbounds %a[%i] : (!llvm.ptr, i64) -> !llvm.ptr, i32
  %v = llvm.load %p : !llvm.ptr -> vector<4xi32>
  %acc_next = llvm.add %acc, %v : vector<4xi32>
  %i_next = llvm.add %i, %c4 : i64
  %cmp = llvm.icmp "slt" %i_next, %n : i64
  llvm.cond_br %cmp, ^bb1(%i_next, %acc_next : i64, vector<4xi32>), ^exit(%acc_next : vector<4xi32>)

^exit(%r: vector<4xi32>):
  %sum = "neura.vector.reduce.add.global"(%r) : (vector<4xi32>) -> i32
  return %sum : i32
}

// MAPPING: "neura.vector.reduce.add.global"({{.*}}) <{reduce_id = 0 : i32}> {dfg_id = {{[0-9]+}} : i32, mapping_locs = [{{{.*}}resource = "tile"

// YAML:      - opcode: "VECTOR.REDUCE.ADD.GLOBAL"
// YAML-NEXT:   id: {{[0-9]+}}
// YAML-NEXT:   reduce_id: 0
// YAML-NEXT:   time_step:
