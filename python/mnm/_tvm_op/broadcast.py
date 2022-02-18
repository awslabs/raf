# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Schedule registries for broadcast operators."""
from .._lib import _reg

_reg.register_broadcast_schedule("mnm.op.tvm.add")
_reg.register_broadcast_schedule("mnm.op.tvm.subtract")
_reg.register_broadcast_schedule("mnm.op.tvm.multiply")
_reg.register_broadcast_schedule("mnm.op.tvm.divide")
_reg.register_broadcast_schedule("mnm.op.tvm.floor_divide")
_reg.register_broadcast_schedule("mnm.op.tvm.maximum")
_reg.register_broadcast_schedule("mnm.op.tvm.minimum")
_reg.register_broadcast_schedule("mnm.op.tvm.bias_add")
_reg.register_broadcast_schedule("mnm.op.tvm.power")
_reg.register_broadcast_schedule("mnm.op.tvm.where")
_reg.register_broadcast_schedule("mnm.op.tvm.logical_and")
_reg.register_broadcast_schedule("mnm.op.tvm.right_shift")
_reg.register_broadcast_schedule("mnm.op.tvm.left_shift")
_reg.register_broadcast_schedule("mnm.op.tvm.equal")
_reg.register_broadcast_schedule("mnm.op.tvm.not_equal")
_reg.register_broadcast_schedule("mnm.op.tvm.less")
_reg.register_broadcast_schedule("mnm.op.tvm.less_equal")
_reg.register_broadcast_schedule("mnm.op.tvm.greater")
_reg.register_broadcast_schedule("mnm.op.tvm.greater_equal")
