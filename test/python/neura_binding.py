# RUN: %python %s | FileCheck %s

from neura_mlir.dialects import func, neura
from neura_mlir.ir import (
    Context,
    DictAttr,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Location,
    Module,
    StringAttr,
    Type,
)


def build_placement(x, y, i32):
    return DictAttr.get(
        {
            "x": IntegerAttr.get(i32, x),
            "y": IntegerAttr.get(i32, y),
        }
    )


def build_module():
    with Context(), Location.unknown():
        neura.register_dialect()

        module = Module.create()
        i32 = IntegerType.get_signless(32)

        # Temporary until the custom Python binding for PredicatedValue exists.
        data_i32 = Type.parse("!neura.data<i32, i1>")

        with InsertionPoint(module.body):
            function = func.FuncOp("add_constant", ([], []))
            entry_block = function.add_entry_block()

        with InsertionPoint(entry_block):
            kernel = neura.KernelOp(
                [],
                [],
                [],
                accelerator=StringAttr.get("neura"),
            )
            kernel_block = kernel.body.blocks.append()
            func.ReturnOp([])

        with InsertionPoint(kernel_block):
            lhs = neura.ConstantOp(
                data_i32,
                IntegerAttr.get(i32, 1),
            )
            lhs.operation.attributes["placement"] = build_placement(0, 0, i32)

            rhs = neura.ConstantOp(
                data_i32,
                IntegerAttr.get(i32, 2),
            )
            rhs.operation.attributes["placement"] = build_placement(2, 0, i32)

            result = neura.AddOp(
                data_i32,
                lhs.result,
                rhs=rhs.result,
            )
            result.operation.attributes["placement"] = build_placement(1, 0, i32)

            neura.YieldOp([], [])

        assert module.operation.verify()
        return module


# CHECK-LABEL: module {
# CHECK: func.func @add_constant()
# CHECK: neura.kernel attributes {accelerator = "neura"}
# CHECK: %[[LHS:.*]] = "neura.constant"()
# CHECK-SAME: value = 1 : i32
# CHECK-SAME: placement = {x = 0 : i32, y = 0 : i32}
# CHECK: %[[RHS:.*]] = "neura.constant"()
# CHECK-SAME: value = 2 : i32
# CHECK-SAME: placement = {x = 2 : i32, y = 0 : i32}
# CHECK: "neura.add"(%[[LHS]], %[[RHS]])
# CHECK-SAME: placement = {x = 1 : i32, y = 0 : i32}
# CHECK: neura.yield
# CHECK: return

print(build_module())
