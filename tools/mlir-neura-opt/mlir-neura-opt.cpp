// tools/mlir-neura-opt/mlir-neura-opt.cpp

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"

#include "Conversion/NeuraConversionPasses.h"
#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraPasses.h"
#include "NeuraDialect/Util/ArchParser.h"
using mlir::neura::Architecture;
using mlir::neura::util::ArchParser;

// Global variable to store architecture spec file path
static std::string architecture_spec_file;

// Global variable to store latency spec file path
static std::string latency_spec_file;

const Architecture &mlir::neura::getArchitecture() {
  static Architecture instance = []() {
    auto arch_parser = ArchParser(architecture_spec_file);
    auto architecture_result = arch_parser.getArchitecture();
    if (failed(architecture_result)) {
      llvm::report_fatal_error("[neura-compiler] Failed to get architecture.");
    }
    return std::move(architecture_result.value());
  }();
  return instance;
}

const std::string &mlir::neura::getLatencySpecFile() {
  return latency_spec_file;
}

int main(int argc, char **argv) {
  // Manually scan and strip --architecture-spec from argv, keep others for
  // MlirOptMain.
  std::vector<char *> forwarded_args;
  forwarded_args.reserve(argc);
  forwarded_args.push_back(argv[0]);
  for (int i = 1; i < argc; ++i) {
    llvm::StringRef arg_ref(argv[i]);
    if (arg_ref == "--architecture-spec") {
      if (i + 1 < argc) {
        architecture_spec_file = argv[i + 1];
        ++i; // skip value
        continue;
      } else {
        llvm::errs() << "[mlir-neura-opt] Error: --architecture-spec option "
                        "requires a value\n";
        return EXIT_FAILURE;
      }
    } else if (arg_ref.starts_with("--architecture-spec=")) {
      architecture_spec_file =
          arg_ref.substr(strlen("--architecture-spec=")).str();
      continue;
    } else if (arg_ref == "--latency-spec") {
      if (i + 1 < argc) {
        latency_spec_file = argv[i + 1];
        ++i; // skip value
        continue;
      }
    } else if (arg_ref.starts_with("--latency-spec=")) {
      latency_spec_file = arg_ref.substr(strlen("--latency-spec=")).str();
      continue;
    }
    forwarded_args.push_back(argv[i]);
  }

  int new_argc = static_cast<int>(forwarded_args.size());
  char **new_argv = forwarded_args.data();

  // Registers MLIR dialects.
  mlir::DialectRegistry registry;
  registry.insert<mlir::neura::NeuraDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::DLTIDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::ml_program::MLProgramDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::linalg::LinalgDialect>();
  registry.insert<mlir::tosa::TosaDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::math::MathDialect>();
  mlir::registerAllExtensions(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);

  mlir::neura::registerPasses();
  mlir::registerAllPasses();
  mlir::registerPasses();
  mlir::registerViewOpGraphPass();

  // Register all standard conversion passes
  mlir::registerConversionPasses();

  // Print architecture spec file info
  if (!architecture_spec_file.empty()) {
    llvm::errs() << "[mlir-neura-opt] Architecture specification file: "
                 << architecture_spec_file << "\n";
  } else {
    llvm::errs() << "[mlir-neura-opt] No architecture specification file "
                    "provided, using default configuration\n";
  }

  // Runs the MLIR optimizer.
  return mlir::asMainReturnCode(mlir::MlirOptMain(
      new_argc, new_argv, "Neura Dialect Optimizer", registry));
}
