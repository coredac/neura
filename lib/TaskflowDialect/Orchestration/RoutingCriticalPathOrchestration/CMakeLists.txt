add_mlir_library(MLIRTaskflowRoutingCriticalPathOrchestration
    RoutingCriticalPathOrchestration.cpp

    DEPENDS
    MLIRTaskflowTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    MLIRFuncDialect
    MLIRTaskflow
    MLIRTaskflowOrchestrationUtil
    LLVMSupport
)
