//===- Orchestration.h - Abstract base class for CGRA orchestration
//--------===//
//
// Defines the abstract Orchestration interface for mapping Taskflow tasks onto
// a 2D multi-CGRA grid.  Concrete strategies derive from this class and
// override runOrchestration().
//
// Modelled after include/NeuraDialect/Mapping/Mapping.h.
//
//===----------------------------------------------------------------------===//

#ifndef TASKFLOW_ORCHESTRATION_H
#define TASKFLOW_ORCHESTRATION_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <string>

namespace mlir {
namespace taskflow {

//===----------------------------------------------------------------------===//
// Orchestration — abstract base class
//===----------------------------------------------------------------------===//

/// Abstract base class for different CGRA task-orchestration strategies.
///
/// Subclasses implement runOrchestration() to map every taskflow.task operation
/// inside `func` onto the physical 2D multi-CGRA grid.  The pass delegates to
/// whichever concrete strategy is installed, making it straightforward to swap
/// in alternative algorithms (e.g. ILP-based, simulated-annealing, etc.)
/// without touching the pass infrastructure.
class Orchestration {
public:
  virtual ~Orchestration() = default;

  /// Runs the orchestration strategy on `func`, annotating each
  /// taskflow.task op with a `task_orchestration_info` attribute that records
  /// the assigned CGRA positions and SRAM locations.
  ///
  /// Returns true on success, false if no valid placement could be found
  /// (e.g. the grid is too full).
  virtual bool runOrchestration(mlir::func::FuncOp func) = 0;

  /// Returns a human-readable name for this strategy (used in log messages).
  virtual std::string getName() const = 0;
};

} // namespace taskflow
} // namespace mlir

#endif // TASKFLOW_ORCHESTRATION_H
