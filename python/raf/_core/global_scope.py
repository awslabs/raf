# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-class-docstring,missing-function-docstring
"""Provide a thread-local scope to cache items"""
import contextlib
import threading


class GlobalScope:
    def __init__(self):
        super(GlobalScope, self).__init__()
        self.storage = threading.local()
        self.storage.scopes = dict()

    @contextlib.contextmanager
    def with_scope(self, item):
        cls = item.__class__
        scopes = self.storage.scopes
        if cls not in scopes:
            scopes[cls] = []
        try:
            scopes[cls].append(item)
            yield
        finally:
            scopes[cls].pop()

    def last(self, cls, default):
        scopes = self.storage.scopes
        if cls not in scopes:
            return default
        if not scopes[cls]:
            return default
        return scopes[cls][-1]


SCOPE = GlobalScope()
