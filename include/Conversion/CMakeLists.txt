set(LLVM_TARGET_DEFINITIONS ConversionPasses.td)
mlir_tablegen(ConversionPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRNeuraConversionIncGen)