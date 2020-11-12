"""Schedule registries for broadcast operators."""
from .._lib import _reg

_reg.register_broadcast_schedule("mnm.op.add")
_reg.register_broadcast_schedule("mnm.op.subtract")
_reg.register_broadcast_schedule("mnm.op.multiply")
_reg.register_broadcast_schedule("mnm.op.divide")
_reg.register_broadcast_schedule("mnm.op.greater")
_reg.register_broadcast_schedule("mnm.op.maximum")
_reg.register_broadcast_schedule("mnm.op.minimum")
_reg.register_broadcast_schedule("mnm.op.bias_add")
_reg.register_broadcast_schedule("mnm.op.power")
