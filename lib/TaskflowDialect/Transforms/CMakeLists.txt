get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_library(MLIRTaskflowTransforms
    OrchestrateTasksOnAcceleratorsPass.cpp
    # NOTE: ImportJointMappingPass now lives in Optimizations/ (compiled into
    # MLIRTaskflowOptimization) so its v2 fusion path can link fuseTaskGroup /
    # UtilizationFuser from ResourceAwareTaskOptimizationPass in-archive.
    ConstructHyperblockFromTaskPass.cpp
    ClassifyTaskAndCounterPass.cpp
    FuseTaskPass.cpp

    DEPENDS
    MLIRTaskflowTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRTaskflow
    MLIRTaskflowOrchestration
    MLIRNeura
    MLIRNeuraTransforms
    MLIRConversion
    ${dialect_libs}
    LLVMSupport
)

add_subdirectory(Optimizations)