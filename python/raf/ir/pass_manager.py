# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pass manager instances."""
# pylint: disable=missing-class-docstring, missing-function-docstring
# pylint: disable=too-few-public-methods, too-many-arguments

from tvm.ir.transform import Pass
from raf._core.core_utils import register_node
from raf._ffi import pass_


@register_node("raf.pass_.RAFSequential")
class RAFSequential(Pass):
    """A pass that works on a sequence of pass objects. Multiple passes can be
    executed sequentially using this class.

    Note that users can also provide a series of passes that they don't want to
    apply when running a sequential pass. Pass dependency will be resolved in
    the backend as well.

    Parameters
    ----------
    passes : Optional[List[Pass]]
        A sequence of passes candidate for optimization.

    opt_level : Optional[int]
        The optimization level of this sequential pass.
        The opt_level of a default sequential pass is set to 2.
        Note that some of the passes within the Sequantial may still not be executed
        if their opt_level is higher than the provided opt_level.

    name : Optional[str]
        The name of the sequential pass.

    required : Optional[List[str]]
        The list of passes that the sequential pass is dependent on.
    """

    def __init__(self, passes=None, opt_level=2, name="sequential", required=None):
        passes = passes if passes else []
        if not isinstance(passes, (list, tuple)):
            raise TypeError("passes must be a list of Pass objects.")

        required = required if required else []
        if not isinstance(required, (list, tuple)):
            raise TypeError("Required is expected to be the type of list/tuple.")

        self.__init_handle_by_constructor__(pass_.RAFSequential, passes, opt_level, name, required)


@register_node("raf.pass_.RAFFunctionPass")
class RAFFunctionPass(Pass):
    """A pass that works on each tvm.relay.Function in a module."""
