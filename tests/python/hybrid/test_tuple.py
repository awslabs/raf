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

from mnm import hybrid


@hybrid
def tuple1():
    a = (1, 2, 3)
    b, c, d = a
    return b + c + d


@hybrid
def tuple2():
    a = ((1, 2), 3)
    (b, c), d = a
    return b + c + d


@hybrid
def tuple3():
    a = ((1, 2), 3)
    b, c = a
    d, e = b
    return d + e + c


def tuple4():
    a = (1, 2, 3)
    b = a[0:2]
    c = b[0:1]
    d = c[0]
    return d


def test_tuple_1():
    assert tuple1() == 6


def test_tuple_2():
    assert tuple2() == 6


def test_tuple_3():
    assert tuple3() == 6


def test_tuple_4():
    assert tuple4() == 1


if __name__ == "__main__":
    test_tuple_1()
    test_tuple_2()
    test_tuple_3()
    test_tuple_4()
