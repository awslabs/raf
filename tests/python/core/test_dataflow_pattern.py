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

# pylint: disable=invalid-name, protected-access
import pytest
import mnm


def test_match_constant():
    c = mnm.ir.const(1)
    value = mnm._core.value.IntValue(1)
    pat = mnm.ir.dataflow_pattern.is_constant(value)
    assert mnm.ir.dataflow_pattern.match(pat, c)


def test_no_match_constant():
    c = mnm.ir.const(1.0)
    value = mnm._core.value.IntValue(1)
    pat = mnm.ir.dataflow_pattern.is_constant(value)
    assert not mnm.ir.dataflow_pattern.match(pat, c)


if __name__ == "__main__":
    pytest.main([__file__])
