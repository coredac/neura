set(LLVM_TARGET_DEFINITIONS NeuraConversionPasses.td)
mlir_tablegen(NeuraConversionPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRNeuraConversionIncGen)
