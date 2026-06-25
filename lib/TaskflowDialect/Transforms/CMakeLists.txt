get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_library(MLIRTaskflowTransforms
    AnalyzeTaskPipelineIntervalPass.cpp
    OrchestrateTaskOnCgraPass.cpp
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
