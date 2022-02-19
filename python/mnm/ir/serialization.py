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

"""IR serialization."""
from mnm._ffi.ir.serialization import SaveJSON, LoadJSON


def save_json(node):
    """Save object as json string. It takes care of extended IR.

    Parameters
    ----------
    node : Object
        A TVM object to be saved.

    Returns
    -------
    json_str : str
        Saved json string.
    """
    return SaveJSON(node)


def load_json(json):
    """Load json string into object. It takes care of extended IR.

    Parameters
    ----------
    json: str
        Saved json string.

    Returns
    -------
    node : Object
        A loaded TVM object
    """
    return LoadJSON(json)
