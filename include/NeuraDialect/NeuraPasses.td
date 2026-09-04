// NeuraPasses.td - Passes for the Neura dialect

#ifndef NEURA_PASSES_TD
#define NEURA_PASSES_TD

include "mlir/Pass/PassBase.td"

//=========================================================//
// Passes for the Neura dialect
//=========================================================//
def AssignAccelerator : Pass<"assign-accelerator", "ModuleOp"> {
  let summary = "Assigns accelerators to operations in the Neura dialect";
  let description = [{Tags non - main functions as neura.kernel.}];
  let constructor = "neura::createAssignAcceleratorPass()";
}

def FusePattern : Pass<"fuse-pattern", "ModuleOp"> {
  let summary = "Fuses patterns in the Neura dialect";
  let description = [{Apply Neura fusion patterns.}];
  let constructor = "neura::createFusePatternPass()";
}

def InsertDataMov : Pass<"insert-data-mov", "ModuleOp"> {
  let summary = "Inserts data move operations in the Neura dialect";
  let description =
      [{Insert neura.data_mov before all neura dialect operations.}];
  let constructor = "neura::createInsertDataMovPass()";
}

def InsertCtrlMov : Pass<"insert-ctrl-mov", "ModuleOp"> {
  let summary = "Inserts ctrl move operations in the Neura dialect";
  let description =
      [{Insert neura.ctrl_mov before all neura dialect operations.}];
  let constructor = "neura::createInsertCtrlMovPass()";
}

def TransformCtrlToDataFlow : Pass<"transform-ctrl-to-data-flow", "ModuleOp"> {
  let summary = "Inserts ctrl move operations in the Neura dialect";
  let description = [{Transform ctrl to predicate-based data flow.}];
  let constructor = "neura::createTransformCtrlToDataFlowPass()";
}

def LeveragePredicatedValue : Pass<"leverage-predicated-value", "ModuleOp"> {
  let summary = "Convert values to predicated values in Neura dialect";
  let description = [{
    This pass converts regular values to predicated values in Neura dialect operations.
    Each value is wrapped in a predicated value type with a default true predicate.
  }];
  let constructor = "neura::createLeveragePredicatedValuePass()";
}

def MapToAccelerator : Pass<"map-to-accelerator", "ModuleOp"> {
  let summary = "Map Neura operations onto a given accelerator";
  let description = [{
    This pass performs mapping from Neura operations to accelerator.

    x-tiles and y-tiles specify the **tile** dimensions of the target array
    (not the CGRA count).  Each CGRA contains a per_cgra_rows × per_cgra_cols
    tile grid (currently 4×4).  So for a single CGRA, x-tiles=4 y-tiles=4;
    for a 1×2 rectangular pair, x-tiles=8 y-tiles=4; etc.

    When x-tiles=0 and y-tiles=0 (the default), the global Architecture
    singleton determines the tile grid — this is equivalent to a single CGRA.

    Examples:
      Single CGRA (default):
        --map-to-accelerator
      1×3 rectangular (3 CGRAs in a row):
        --map-to-accelerator x-tiles=12 y-tiles=4
      T-shape (4 CGRAs: top row of 3 + centre below):
        --map-to-accelerator x-tiles=12 y-tiles=8 \
            valid-tiles="0_0,1_0,2_0,3_0,4_0,5_0,6_0,7_0,8_0,9_0,10_0,11_0,\
                         4_1,5_1,6_1,7_1,4_4,5_4,6_4,7_4,4_5,5_5,6_5,7_5"
  }];
  let options =
      [Option<
           "x_tiles", "x-tiles", "int", /*default=*/"0",
           "Total number of tiles in the X dimension of the target array "
           "(not the number of CGRAs).  Each CGRA contributes per_cgra_cols "
           "tiles.  0 means use the global Architecture singleton (1 CGRA).">,
       Option<
           "y_tiles", "y-tiles", "int", /*default=*/"0",
           "Total number of tiles in the Y dimension of the target array "
           "(not the number of CGRAs).  Each CGRA contributes per_cgra_rows "
           "tiles.  0 means use the global Architecture singleton (1 CGRA).">,
       Option<
           "valid_tiles", "valid-tiles", "std::string", /*default=*/"\"\"",
           "Comma-separated list of tile coordinates (x_y) that are actually "
           "present in the array, used for non-rectangular CGRA shapes such as "
           "L-blocks or T-blocks. Empty string means all tiles in the "
           "x-tiles x y-tiles rectangle are valid. "
           "Example: 0_0,1_0,0_1 selects three tiles forming an L-shape.">];
  let constructor = "neura::createMapToAcceleratorPass()";
}

def GenerateCode : Pass<"generate-code", "ModuleOp"> {
  let summary = "Generate JSON-formatted instructions from mapped Neura IR";
  let description = [{
    This pass generates JSON file containing the instructions.
    The instructions can be encoded into configuration signals.
  }];
  let constructor = "neura::createGenerateCodePass()";
}

def FuseLoopControl : Pass<"fuse-loop-control", "ModuleOp"> {
  let summary = "Fuses loop control operations in the Neura dialect";
  let description = [{
    This pass fuses loop control operations.
  }];
  let constructor = "neura::createFuseLoopControlPass()";
}

def CanonicalizeReturn : Pass<"canonicalize-return", "ModuleOp"> {
  let summary = "Adds execution conditions to void return operations";
  let description = [{
    This pass processes void functions and adds trigger conditions to their
    return operations. For void functions, the return operation needs an
    execution condition to properly trigger in dataflow mode.
    
    This pass should run before CanonicalizeLiveIn and TransformCtrlToDataFlow.
  }];
  let constructor = "neura::createCanonicalizeReturnPass()";
}

def CanonicalizeLiveIn : Pass<"canonicalize-live-in", "ModuleOp"> {
  let summary = "Canonicalizes live-in values/operations in each basic block.";
  let description = [{
    This pass applies canonicalization transformations to Neura dialect operations.
    The canonicalization includes:
    1. Converting all live-in values of each basic block to block arguments.
  }];
  let constructor = "neura::createCanonicalizeLiveInPass()";
}

def PromoteInputArgToConst : Pass<"promote-input-arg-to-const", "ModuleOp"> {
  let summary = "Promotes input arguments of functions or neura.kernels to "
                "neura constant operations";
  let description = [{
    This pass promotes input arguments of functions or neura.kernels to
    neura.constant operations. For template kernels, arguments explicitly bound
    to input ports remain block arguments so that TemplateMapping can inject
    them through TileArray boundary ports.
  }];
  let constructor = "neura::createPromoteInputArgToConstPass()";
}

def CanonicalizeCast : Pass<"canonicalize-cast", "ModuleOp"> {
  let summary = "Canonicalizes cast operations in the Neura dialect";
  let description = [{
    This pass applies canonicalization transformations to neura::cast operations.
    The canonicalization includes:
    1. Removing redundant casts.
    2. Converting index (i64) types to i64 (index).
  }];
  let constructor = "neura::createCanonicalizeCastPass()";
}

def FoldConstant : Pass<"fold-constant", "ModuleOp"> {
  let summary = "Folds constant operations in the Neura dialect";
  let description = [{
    This pass applies constant folding transformations to Neura dialect operations.
    The folding includes:
    1. Folding constant operations.
    2. Removing dead operations.
  }];
  let constructor = "neura::createFoldConstantPass()";
}

def TransformToSteerControl
    : Pass<"transform-to-steer-control", "func::FuncOp"> {
  let summary = "Transform control flow into data flow using steer control";
  let description = [{
    This pass transforms Neura control flow graphs (CDFG) into pure dataflow graphs (DFG)
    using steer control operations like true_steer, false_steer, carry, and merge.
    Unlike predication-based approaches, steer control explicitly directs data through
    different paths based on conditions.
  }];
  let constructor = "neura::createTransformToSteerControlPass()";
}

def RemovePredicatedType : Pass<"remove-predicated-type", "ModuleOp"> {
  let summary = "Removes predicated types from Neura dialect operations";
  let description = [{
    This pass removes predicated types from Neura dialect operations,
    converting them back to regular types.
  }];
  let constructor = "neura::createRemovePredicatedTypePass()";
}

def IterMergePattern : Pass<"iter-merge-pattern", "ModuleOp"> {
  let summary = "Iteratively merge and identify common patterns in DFG";
  let description = [{
    This pass iteratively traverses the DFG (Data Flow Graph) and identifies
    common patterns using graph mining algorithms. It performs iterative merging
    of patterns and stores results for subsequent passes.
  }];
  let constructor = "neura::createIterMergePatternPass()";
}

def InitPattern : Pass<"init-pattern", "ModuleOp"> {
  let summary =
      "Initialize and identify common patterns in DFG (single iteration)";
  let description = [{
    This pass traverses the DFG (Data Flow Graph) and identifies common patterns
    using graph mining algorithms. It performs a single iteration of pattern merging
    with a configurable minimum support threshold.
  }];
  let constructor = "neura::createInitPatternPass()";
}

def HardwareMerge : Pass<"hardware-merge", "ModuleOp"> {
  let summary = "Merge and optimize hardware units for pattern execution";
  let description = [{
    This pass analyzes patterns (fused_op regions) and designs an optimal
    hardware configuration that supports all patterns while minimizing
    hardware cost. It uses a Functional Unit (FU) based design where each
    FU executes exactly one operation type.
    
    Algorithm:
    1. Extract pattern DAGs with topological structure from fused_op regions
    2. Sort patterns by complexity (distinct operation count and cost)
    3. For each pattern, try to accommodate it into existing templates by reusing FUs with matching operation types
    4. If accommodation cost is too high, create a new template with dedicated FUs for the pattern
    5. Generate FU connections based on pattern dependencies with transitive reduction optimization
    6. Generate execution plans with parallel execution stages
    7. Output the final hardware configuration as a JSON file
  }];
  let constructor = "neura::createHardwareMergePass()";
}

def InitExecLatency : Pass<"init-exec-latency", "ModuleOp"> {
  let summary = "Initialize execution latency information";
  let description = [{
    This pass initializes execution latency information.
  }];
  let constructor = "neura::createInitExecLatencyPass()";
}
#endif // NEURA_PASSES_TD