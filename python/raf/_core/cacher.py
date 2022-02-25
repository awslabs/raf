# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import weakref

from .core_utils import bfs


class Cacher:
    def __init__(self):
        super(Cacher, self).__init__()
        self.__enabled = False
        self.__cache = {}
        self.__parents = weakref.WeakKeyDictionary()

    def __delattr__(self, name):
        if getattr(self, "_Cacher__enabled", False):
            del_child(self, getattr(self, name))
        super().__delattr__(name)

    def __setattr__(self, name, value):
        if getattr(self, "_Cacher__enabled", False):
            if hasattr(self, name):
                delattr(self, name)
            add_child(self, value)
        super().__setattr__(name, value)


# pylint: disable=protected-access


def enable(cacher):
    cacher._Cacher__enabled = True


def disable(cacher):
    cacher._Cacher__enabled = False


def get_cache(cacher, key, default_value):
    return cacher._Cacher__cache.get(key, default_value)


def set_cache(cacher, key, value):
    cacher._Cacher__cache[key] = value


def invalidate(root_cacher, *, include_self, recursive):
    if not root_cacher._Cacher__enabled:
        return

    def on_pop(cacher):
        if include_self or (cacher is not root_cacher):
            object.__setattr__(cacher, "_Cacher__cache", {})

    def on_next(cacher):
        nexts = []
        for x in cacher._Cacher__parents.keyrefs():
            x = x()
            if x is not None:
                nexts.append(x)
        return nexts

    bfs([root_cacher], on_pop, on_next, recursive=recursive)


def add_child(cacher, child):
    if isinstance(child, Cacher):
        parents = child._Cacher__parents
        if cacher in parents:
            parents[cacher] += 1
        else:
            parents[cacher] = 1
    invalidate(cacher, include_self=True, recursive=True)


def del_child(cacher, child):
    if isinstance(child, Cacher):
        parents = child._Cacher__parents
        parents[cacher] -= 1
        if parents[cacher] == 0:
            del parents[cacher]
    invalidate(cacher, include_self=True, recursive=True)


# pylint: enable=protected-access
