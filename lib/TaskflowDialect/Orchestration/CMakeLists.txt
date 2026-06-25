add_mlir_library(MLIRTaskflowOrchestrationUtil
    orchestration_utils.cpp

    PARTIAL_SOURCES_INTENDED

    DEPENDS
    MLIRTaskflowTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRFuncDialect
    MLIRTaskflow
    LLVMSupport
)

add_subdirectory(RoutingCriticalPathOrchestration)
add_subdirectory(ThroughputGuidedTaskOrchestration)

add_mlir_library(MLIRTaskflowOrchestration
    Orchestration.cpp

    PARTIAL_SOURCES_INTENDED

    DEPENDS
    MLIRTaskflowTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRTaskflowOrchestrationUtil
    MLIRTaskflowRoutingCriticalPathOrchestration
    MLIRTaskflowThroughputGuidedTaskOrchestration
)
