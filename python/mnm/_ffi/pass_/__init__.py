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

"""Auto generated. Do not touch."""
# pylint: disable=redefined-builtin,line-too-long
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from __future__ import absolute_import
from ._internal import ASAPStreamSchedule
from ._internal import AnnotateCollectiveOps
from ._internal import AnnotateTarget
from ._internal import AssignDevice
from ._internal import AutoCast
from ._internal import AutoDataParallel
from ._internal import AutoDiff
from ._internal import BindParam
from ._internal import CanonicalizeOps
from ._internal import ContextAnalysis
from ._internal import DataParallelSchedule
from ._internal import DeadCodeElimination
from ._internal import Deduplicate
from ._internal import DispatchDialect
from ._internal import EnforceSync
from ._internal import EraseType
from ._internal import EstimateGFLOPS
from ._internal import ExprAppend
from ._internal import ExtractBinding
from ._internal import FlattenClosure
from ._internal import FoldConstant
from ._internal import FromRelay
from ._internal import FuseDialect
from ._internal import FuseTVM
from ._internal import GradientInputSelection
from ._internal import IOSStreamSchedule
from ._internal import InferType
from ._internal import InlineBackward
from ._internal import InlineClosure
from ._internal import InlineLet
from ._internal import InlinePrimitives
from ._internal import InplaceUpdate
from ._internal import LambdaLift
from ._internal import LiftBranchBody
from ._internal import LivenessAnalysis
from ._internal import MNMSequential
from ._internal import ManifestAlloc
from ._internal import MemoryPlan
from ._internal import MemorySchedule
from ._internal import MergeCompilerRegions
from ._internal import PartitionANF
from ._internal import PartitionGradient
from ._internal import PartitionGraph
from ._internal import PrintIR
from ._internal import Rematerialization
from ._internal import RenameVars
from ._internal import SimplifyExpr
from ._internal import Substitute
from ._internal import ToANormalForm
from ._internal import ToBasicBlockNormalForm
from ._internal import ToGraphNormalForm
from ._internal import ValidateInplaceUpdate
from ._internal import WavefrontStreamSchedule
from ._internal import dataflow_pattern_match
from ._internal import dataflow_pattern_partition
from ._internal import dataflow_pattern_rewrite
from ._internal import is_constant
from ._internal import validate_relay_param_name
