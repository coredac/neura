include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_mlir_conversion_library(MLIRNeuraArithToNeuraPass
  ArithToNeuraPass.cpp

  DEPENDS
  MLIRNeuraConversionIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransforms
)