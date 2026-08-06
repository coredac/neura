include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_mlir_conversion_library(MLIRTaskflowOptimization
  AffineLoopTreeSerializationPass.cpp
  AffineLoopPerfectionPass.cpp
  MemoryAccessStreamingFusion.cpp
  ResourceAwareTaskOptimizationPass.cpp
  # Lives in this library (not Transforms) so its v2 fusion path links against
  # fuseTaskGroup / UtilizationFuser in ResourceAwareTaskOptimizationPass within
  # this same archive, avoiding a Transforms<->Optimization link cycle.
  ImportJointMappingPass.cpp

  DEPENDS
  MLIRTaskflowTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTaskflow
  MLIRTaskflowTransforms
  MLIRAffineDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRNeura
  MLIRNeuraTransforms
  MLIRConversion
  MLIRSupport
)