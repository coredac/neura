add_mlir_dialect_library(MLIRNeuraAffineToNeuraPass
  AffineToNeuraPass.cpp
  LoopNestAnalysis.cpp

  DEPENDS
  MLIRNeuraConversionIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRNeura
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRFuncDialect
)
