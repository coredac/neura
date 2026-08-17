#include "Neura-C/Neura.h"

#include "NeuraDialect/NeuraDialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Neura, neura, mlir::neura::NeuraDialect)