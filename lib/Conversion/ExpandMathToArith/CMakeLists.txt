add_mlir_library(MLIRExpandMathToArithPass
  ExpandMathToArithPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Conversion

  DEPENDS
  MLIRConversionIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRArithDialect
  MLIRMathDialect
)
