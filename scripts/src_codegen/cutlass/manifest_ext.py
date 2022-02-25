# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The extended code manifest. Origin: cutlass/tools/library/scripts/manifest.py"""
import os
import shutil

import manifest
import gemm_operation
import conv2d_operation
from library import *

import conv2d_operation_ext
import gemm_operation_ext


class EmitOperationKindLibraryExt(manifest.EmitOperationKindLibrary):
    def __init__(self, generated_path, kind, args):
        super().__init__(generated_path, kind, args)
        self.emitters = {
            OperationKind.Gemm: gemm_operation_ext.EmitGemmConfigurationLibraryExt,
            OperationKind.Conv2d: conv2d_operation_ext.EmitConv2dConfigurationLibraryExt,
        }


def convert(op):
    if isinstance(op, gemm_operation.GemmOperation):
        return gemm_operation_ext.make_gemm_operation_ext(op)
    if isinstance(op, conv2d_operation.Conv2dOperation):
        return conv2d_operation_ext.make_conv2d_operation_ext(op)
    return op


class ManifestExt(manifest.Manifest):
    def append(self, operation):
        op = convert(operation)
        super().append(op)

    def emit(self, target=GeneratorTarget.Library):

        operation_emitters = {GeneratorTarget.Library: EmitOperationKindLibraryExt}

        generated_path = os.path.join(self.args.curr_build_dir, "generated")

        # create generated/
        if os.path.exists(generated_path):
            shutil.rmtree(generated_path)

        os.mkdir(generated_path)

        source_files = []

        top_level_path = os.path.join(generated_path, "initialize_all.cpp")
        with open(top_level_path, "w") as top_level_file:

            if target == GeneratorTarget.Library:
                source_files.append(top_level_path)

            prototypes = []
            for operation_kind, configurations in self.operations.items():
                prototypes.append(
                    SubstituteTemplate(
                        "void initialize_all_${operation_kind}_operations(Manifest &manifest);",
                        {"operation_kind": OperationKindNames[operation_kind]},
                    )
                )

            top_level_file.write(
                SubstituteTemplate(self.top_level_prologue, {"prototypes": "\n".join(prototypes)})
            )

            top_level_file.write(
                SubstituteTemplate(
                    self.top_level_reserve, {"operation_count": str(self.operation_count)}
                )
            )

            # for each operation kind, emit initializer for all configurations
            for operation_kind, configurations in self.operations.items():

                with operation_emitters[target](
                    generated_path, operation_kind, self.args
                ) as operation_kind_emitter:
                    for configuration_name, operations in configurations.items():
                        operation_kind_emitter.emit(configuration_name, operations)

                    source_files += operation_kind_emitter.source_files

                top_level_file.write(
                    SubstituteTemplate(
                        "  initialize_all_${operation_kind}_operations(manifest);\n",
                        {"operation_kind": OperationKindNames[operation_kind]},
                    )
                )

            top_level_file.write(self.top_level_epilogue)

        # write the manifest.cmake file containing paths from all targets
        manifest_path = os.path.join(generated_path, "manifest.cmake")
        with open(manifest_path, "w") as manifest_file:

            target_name = "cutlass_library_objs"

            target_text = SubstituteTemplate(
                """cutlass_target_sources(
  ${target_name}
  BATCH_SOURCES ON
  PRIVATE
""",
                {"target_name": target_name},
            )

            manifest_file.write(target_text)

            for source_file in source_files:
                manifest_file.write("    %s\n" % str(source_file.replace("\\", "/")))
            manifest_file.write(")")


def make_manifest_ext(manifest):
    res = ManifestExt(manifest.args)
    for op in manifest.operations_by_name.values():
        res.append(convert(op))
    return res
