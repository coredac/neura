// neura-compiler.cpp
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Conversion/NeuraConversionPasses.h"
#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraPasses.h"
#include "NeuraDialect/Util/ArchParser.h"
#include "mlir/Support/LogicalResult.h"

using mlir::neura::Architecture;
using mlir::neura::util::ArchParser;

// Global variable to store architecture spec file path
static std::string architecture_spec_file;

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
        llvm::errs() << "[neura-compiler] Error: --architecture-spec option "
                        "requires a value\n";
        return EXIT_FAILURE;
      }
    } else if (arg_ref.starts_with("--architecture-spec=")) {
      architecture_spec_file =
          arg_ref.substr(strlen("--architecture-spec=")).str();
      continue;
    }
    forwarded_args.push_back(argv[i]);
  }

  int new_argc = static_cast<int>(forwarded_args.size());
  char **new_argv = forwarded_args.data();
  // Registers MLIR dialects.
  mlir::DialectRegistry registry;
  registry.insert<mlir::neura::NeuraDialect>();

  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);

  mlir::neura::registerNeuraConversionPassPipeline();

  // Print architecture spec file info
  if (!architecture_spec_file.empty()) {
    llvm::errs() << "[neura-compiler] Architecture specification file: "
                 << architecture_spec_file << "\n";
  } else {
    llvm::errs() << "[neura-compiler] No architecture specification file "
                    "provided, using default configuration\n";
  }
  // Runs the MLIR optimizer.
  return mlir::asMainReturnCode(mlir::MlirOptMain(
      new_argc, new_argv, "Neura Dialect Compiler", registry));
}
