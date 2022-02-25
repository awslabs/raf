# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The extended generator. Origin: cutlass/tools/library/scripts/generator.py"""
import argparse

import generator
from generator import CreateGemmOperator, CreateConv2dOperator
from library import *
from manifest import *

import manifest_ext
from library_ext import *


def GenerateSM50_Simt_Epilogue(manifest, args):
    """Extention to raf/3rdparty/cutlass/tools/library/scripts/generator.py::GenerateSM50_Simt"""
    layouts = [
        (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
        (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
        (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
        (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    ]

    math_instructions = [
        MathInstruction(
            [1, 1, 1],
            DataType.f32,
            DataType.f32,
            DataType.f32,
            OpcodeClass.Simt,
            MathOperation.multiply_add,
        ),
        MathInstruction(
            [1, 1, 1],
            DataType.f64,
            DataType.f64,
            DataType.f64,
            OpcodeClass.Simt,
            MathOperation.multiply_add,
        ),
    ]

    # compute capability requirement for these kernels:
    # 50 <= compute capability
    min_cc = 50
    max_cc = 1024

    alignment_constraints = [
        1,
    ]

    for math_inst in math_instructions:
        tile_descriptions = [
            TileDescription([128, 128, 8], 2, [4, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([128, 64, 8], 2, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([64, 128, 8], 2, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([64, 64, 8], 2, [2, 1, 1], math_inst, min_cc, max_cc),
            TileDescription([128, 32, 8], 2, [2, 1, 1], math_inst, min_cc, max_cc),
            TileDescription([32, 128, 8], 2, [1, 2, 1], math_inst, min_cc, max_cc),
        ]

        data_type = [
            math_inst.element_a,
            math_inst.element_b,
            math_inst.element_accumulator,
            math_inst.element_accumulator,
        ]

        CreateGemmOperator(
            manifest,
            layouts,
            tile_descriptions,
            data_type,
            alignment_constraints,
            epilogue_functor=EpilogueFunctorExt.LinearCombinationRelu,
        )
        CreateGemmOperator(
            manifest,
            layouts,
            tile_descriptions,
            data_type,
            alignment_constraints,
            epilogue_functor=EpilogueFunctorExt.LinearCombinationGELU,
        )

        if math_inst.element_a == DataType.f32:
            conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
            CreateConv2dOperator(
                manifest,
                conv_layout,
                tile_descriptions,
                data_type,
                1,
                epilogue_functor=EpilogueFunctorExt.LinearCombinationRelu,
            )


def GenerateSM50(manifest, args):
    GenerateSM50_Simt_Epilogue(manifest, args)


def GenerateSM70_TensorOp_884_Epilogue(manifest, args):

    if not generator.CudaToolkitVersionSatisfies(args.cuda_version, 10, 1):
        return

    layouts = [
        (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
        (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
        (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
        (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    ]

    math_instructions = [
        MathInstruction(
            [8, 8, 4],
            DataType.f16,
            DataType.f16,
            DataType.f32,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add,
        ),
        MathInstruction(
            [8, 8, 4],
            DataType.f16,
            DataType.f16,
            DataType.f16,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add,
        ),
    ]

    min_cc = 70
    max_cc = 75

    alignment_constraints = [8, 4, 2, 1]

    for math_inst in math_instructions:
        tile_descriptions = [
            TileDescription([256, 128, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([128, 256, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc),
            TileDescription([128, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([64, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([128, 64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([64, 64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
        ]

        data_type = [
            math_inst.element_a,
            math_inst.element_b,
            math_inst.element_accumulator,
            math_inst.element_accumulator,
        ]

        CreateGemmOperator(
            manifest,
            layouts,
            tile_descriptions,
            data_type,
            alignment_constraints,
            epilogue_functor=EpilogueFunctorExt.LinearCombinationRelu,
        )
        CreateGemmOperator(
            manifest,
            layouts,
            tile_descriptions,
            data_type,
            alignment_constraints,
            epilogue_functor=EpilogueFunctorExt.LinearCombinationGELU,
        )

        conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
        # CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, 8)

        # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
        if math_inst.element_a != math_inst.element_accumulator:

            data_type_mixed = [
                math_inst.element_a,
                math_inst.element_b,
                math_inst.element_a,
                math_inst.element_accumulator,
            ]

            CreateGemmOperator(
                manifest,
                layouts,
                tile_descriptions,
                data_type_mixed,
                alignment_constraints,
                epilogue_functor=EpilogueFunctorExt.LinearCombinationRelu,
            )
            CreateGemmOperator(
                manifest,
                layouts,
                tile_descriptions,
                data_type_mixed,
                alignment_constraints,
                epilogue_functor=EpilogueFunctorExt.LinearCombinationGELU,
            )

        #   CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type_mixed, 8)


def GenerateSM70(manifest, args):
    GenerateSM70_TensorOp_884_Epilogue(manifest, args)


def GenerateSM80_Simt_f32_Epilogue(manifest, args):
    """Extention to raf/3rdparty/cutlass/tools/library/scripts/generator.py::GenerateSM80_Simt"""
    layouts = [
        (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
        (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
        (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
        (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    ]

    math_instructions = [
        MathInstruction(
            [1, 1, 1],
            DataType.f32,
            DataType.f32,
            DataType.f32,
            OpcodeClass.Simt,
            MathOperation.multiply_add,
        ),
    ]

    # compute capability requirement for these kernels:
    # 80 <= compute capability
    min_cc = 80
    max_cc = 1024

    alignment_constraints = [
        1,
    ]

    for math_inst in math_instructions:
        tile_descriptions = [
            TileDescription([256, 128, 8], 5, [4, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([128, 256, 8], 5, [2, 4, 1], math_inst, min_cc, max_cc),
            TileDescription([128, 128, 8], 5, [4, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([256, 128, 8], 4, [4, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([128, 256, 8], 4, [2, 4, 1], math_inst, min_cc, max_cc),
            TileDescription([128, 128, 8], 4, [4, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([128, 64, 8], 5, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([64, 128, 8], 5, [2, 2, 1], math_inst, min_cc, max_cc),
            TileDescription([64, 64, 8], 5, [2, 1, 1], math_inst, min_cc, max_cc),
            TileDescription([128, 32, 8], 5, [2, 1, 1], math_inst, min_cc, max_cc),
            TileDescription([32, 128, 8], 5, [1, 2, 1], math_inst, min_cc, max_cc),
        ]

        data_type = [
            math_inst.element_a,
            math_inst.element_b,
            math_inst.element_accumulator,
            math_inst.element_accumulator,
        ]

        CreateGemmOperator(
            manifest,
            layouts,
            tile_descriptions,
            data_type,
            alignment_constraints,
            epilogue_functor=EpilogueFunctorExt.LinearCombinationRelu,
        )
        CreateGemmOperator(
            manifest,
            layouts,
            tile_descriptions,
            data_type,
            alignment_constraints,
            epilogue_functor=EpilogueFunctorExt.LinearCombinationGELU,
        )

        conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
        CreateConv2dOperator(
            manifest,
            conv_layout,
            tile_descriptions,
            data_type,
            1,
            epilogue_functor=EpilogueFunctorExt.LinearCombinationRelu,
        )


def GenerateSM80(manifest, args):
    GenerateSM80_Simt_f32_Epilogue(manifest, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates device kernel registration code for CUTLASS Kernels"
    )
    parser.add_argument(
        "--operations", default="all", help="Specifies the operation to generate (gemm, all)"
    )
    parser.add_argument(
        "--build-dir", default=".", required=False, help="CUTLASS top-level build directory"
    )
    parser.add_argument(
        "--curr-build-dir",
        default=".",
        help="CUTLASS current build directory. cmake files will be emitted in this directory",
    )
    parser.add_argument(
        "--generator-target", default="library", help="Target of CUTLASS Library Generator."
    )
    parser.add_argument(
        "--architectures", default="53;60;61;70;75;80", help="Target compute architectures"
    )
    parser.add_argument(
        "--kernels", default="", help="Comma delimited list to filter kernels by name."
    )
    parser.add_argument(
        "--ignore-kernels",
        default="",
        help="Comma delimited list of kernels to exclude from build.",
    )
    parser.add_argument(
        "--cuda-version", default="11.0.0", help="Semantic version string of CUDA Toolkit"
    )
    parser.add_argument(
        "--kernel-filter-file",
        type=str,
        default=None,
        required=False,
        help="Full path of filter file",
    )
    parser.add_argument(
        "--selected-kernel-list",
        type=str,
        default=None,
        required=False,
        help="Specify the output log file containing all enabled kernels in this build",
    )

    args = parser.parse_args()

    manifest = manifest_ext.ManifestExt(args)
    generator.GenerateSM50(manifest, args)
    generator.GenerateSM60(manifest, args)
    generator.GenerateSM61(manifest, args)
    generator.GenerateSM70(manifest, args)
    generator.GenerateSM75(manifest, args)
    generator.GenerateSM80(manifest, args)
    GenerateSM50(manifest, args)
    GenerateSM70(manifest, args)
    GenerateSM80(manifest, args)
    print(manifest.operation_count)

    if "library" in args.generator_target.split(","):
        manifest.emit(GeneratorTarget.Library)

    if args.selected_kernel_list is not None:
        if len(manifest.selected_kernels) > 0:
            with open(args.selected_kernel_list, "w") as file_writer:
                for line in manifest.selected_kernels:
                    file_writer.write("%s\n" % line)
