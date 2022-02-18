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

import os

from .codegen_utils import API


def get(path_prefix="./src/"):
    apis = []

    for root, _, files in os.walk(path_prefix):
        for path in files:
            path = os.path.join(root, path)
            if not (path.endswith(".cc") or path.endswith(".cu")):
                continue
            with open(path, "r") as i_f:
                for lineno, line in enumerate(i_f, 1):
                    line = line.strip()

                    if not line.startswith('MNM_REGISTER_GLOBAL("mnm.'):
                        continue
                    name = line[line.index('("') + 2 : line.index('")')]
                    apis.append(API(name=name, path=path, lineno=lineno))

    return apis
