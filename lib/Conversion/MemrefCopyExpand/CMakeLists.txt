include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_mlir_conversion_library(MLIRNeuraMemrefCopyExpandPass
  ExpandMemrefCopyPass.cpp

  DEPENDS
  MLIRNeuraConversionIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRAffineDialect
    MLIRMemRefDialect
    MLIRArithDialect
)
