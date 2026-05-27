get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_library(MLIRTaskflowTransforms
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
    MLIRTaskflowAllocation
    MLIRNeura
    MLIRNeuraTransforms
    MLIRConversion
    ${dialect_libs}
    LLVMSupport
)

add_subdirectory(Optimizations)