add_mlir_library(MLIRTaskflowOrchestration
    orchestration_utils.cpp
    RoutingCriticalPathOrchestration/RoutingCriticalPathOrchestration.cpp

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
