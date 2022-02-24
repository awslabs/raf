# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
from collections import defaultdict

from . import def_api
from .codegen_utils import write_to_file


def gen_internal_file(apis):
    FILE = """
# pylint: disable=invalid-name,redefined-builtin,line-too-long
# pylint: disable=missing-class-docstring,missing-function-docstring
\"\"\"Auto generated. Do not touch.\"\"\"
from raf._lib import _APIS
{APIs}
""".strip()
    apis = "\n".join(map(gen_api, sorted(apis, key=lambda api: api.name)))
    return FILE.format(APIs=apis)


def gen_api(api):
    API = """
# Defined in {PATH}
{LAST_NAME} = _APIS.get("{FULL_NAME}", None)
""".strip()
    path = api.path
    lineno = api.lineno
    last_name = api.name.split(".")[-1]
    full_name = api.name
    return API.format(PATH=path, LINENO=lineno, LAST_NAME=last_name, FULL_NAME=full_name)


def gen_init_file(apis, dirs):
    FILE = """
\"\"\"Auto generated. Do not touch.\"\"\"
# pylint: disable=redefined-builtin,line-too-long
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from __future__ import absolute_import
{INTERNALS}
{DIRS}
""".strip()
    INTERNAL = """
from ._internal import {NAME}
""".strip()
    DIR = """
from . import {NAME}
""".strip()
    internals = "\n".join(
        INTERNAL.format(NAME=api.name.split(".")[-1])
        for api in sorted(apis, key=lambda api: api.name)
    )
    dirs = "\n".join(DIR.format(NAME=dir) for dir in sorted(dirs))
    return FILE.format(INTERNALS=internals, DIRS=dirs)


def main(path_prefix="./python/raf/_ffi/"):
    api_files = defaultdict(list)
    # collect apis to their corresponding bins
    for api in def_api.get():
        prefix = ".".join(api.name.split(".")[1:-1])
        api_files[prefix].append(api)
    # generate code
    srcs = {prefix: gen_internal_file(apis) for prefix, apis in api_files.items()}
    # generate _internal.py
    if os.path.exists(path_prefix):
        shutil.rmtree(path_prefix)
    for prefix, src in srcs.items():
        path = os.path.join(path_prefix, *prefix.split("."))
        os.makedirs(path, exist_ok=True)
        write_to_file(os.path.join(path, "_internal.py"), src)
    # generate __init__.py
    for prefix, dirs, files in os.walk(path_prefix):
        assert prefix.startswith(path_prefix)
        assert files == ["_internal.py"] or not files
        # path of __init__.py
        path = os.path.join(prefix, "__init__.py")
        # some dirty reverse engineering
        prefix = prefix[len(path_prefix) :]
        if prefix.startswith("/"):
            prefix = prefix[1:]
        if prefix.endswith("/"):
            prefix = prefix[:-1]
        prefix = prefix.replace("/", ".")
        result = gen_init_file(api_files.get(prefix, []), dirs)
        write_to_file(path, result)


if __name__ == "__main__":
    main()
