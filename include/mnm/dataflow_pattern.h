/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file dataflow_pattern.h
 * \brief Dataflow pattern of Meta IR
 */
#pragma once
#include "./ir.h"

namespace mnm {
namespace ir {

// Pattern Nodes
using tvm::relay::DFPattern;
using tvm::relay::DFPatternCallback;
using tvm::relay::DFPatternNode;

using tvm::relay::AltPattern;
using tvm::relay::AltPatternNode;

using tvm::relay::AttrPattern;
using tvm::relay::AttrPatternNode;

using tvm::relay::CallPattern;
using tvm::relay::CallPatternNode;

using tvm::relay::DataTypePattern;
using tvm::relay::DataTypePatternNode;

using tvm::relay::DominatorPattern;
using tvm::relay::DominatorPatternNode;

using tvm::relay::ExprPattern;
using tvm::relay::ExprPatternNode;

using tvm::relay::FunctionPattern;
using tvm::relay::FunctionPatternNode;

using tvm::relay::IfPattern;
using tvm::relay::IfPatternNode;

using tvm::relay::LetPattern;
using tvm::relay::LetPatternNode;

using tvm::relay::ShapePattern;
using tvm::relay::ShapePatternNode;

using tvm::relay::TupleGetItemPattern;
using tvm::relay::TupleGetItemPatternNode;

using tvm::relay::TuplePattern;
using tvm::relay::TuplePatternNode;

using tvm::relay::TypePattern;
using tvm::relay::TypePatternNode;

using tvm::relay::VarPattern;
using tvm::relay::VarPatternNode;

using tvm::relay::WildcardPattern;
using tvm::relay::WildcardPatternNode;

using ConstantPattern = tvm::relay::ConstantPattern;
using RelayConstantPattern = tvm::relay::ConstantPattern;
using RelayConstantPatternNode = tvm::relay::ConstantPatternNode;

// Pattern functors
using tvm::relay::DFPatternFunctor;
using tvm::relay::DFPatternVisitor;

/*! \brief Container for Meta Constant */
class ConstantPatternNode : public RelayConstantPatternNode {
 public:
  /*! \brief The value in the constant pattern node */
  ObjectRef value;
};

// Pattern sugars
/*! \brief Syntatic Sugar for creating a ExprPattern */
using tvm::relay::IsExpr;
/*! \brief Syntatic Sugar for creating a ExprPattern base on an Op*/
using tvm::relay::IsOp;
/*! \brief Syntatic Sugar for creating a TuplePattern*/
using tvm::relay::IsTuple;
/*! \brief Syntatic Sugar for creating a TupleGetItemPattern*/
using tvm::relay::IsTupleGetItem;
/*! \brief Syntatic Sugar for creating a VarPattern with a name */
using tvm::relay::IsVar;
/*! \brief Syntatic Sugar for creating a WildcardPattern */
using tvm::relay::IsWildcard;
/*! \brief Syntatic Sugar for creating a RelayConstantPattern */
DFPattern IsRelayConstant();
/*! \brief Syntatic Sugar for creating a ConstantPattern */
DFPattern IsConstant(ObjectRef value);

// Pattern Utilities
/*!
 * \brief Determine if a pattern matches an expression
 *
 * \param pattern The pattern to match
 * \param expr The expression to match
 *
 * \return Return true if the pattern and the expression match, return false otherwise.
 */
bool MNMMatchPattern(DFPattern pattern, Expr expr);

/*!
 * \brief Rewrite an expression based on some number of DFPatternCallbacks
 *
 * \param callbacks An array of DFPatternCallback Nodes
 * \param expr The expression to rewrite
 * \param mod The module that associates with the expr
 *
 * \return Return An Expr with every match of the pattern inside the callbacks rewritten by the
 * functions inside the callbacks
 */
Expr MNMRewritePatterns(Array<DFPatternCallback> callbacks, Expr expr, IRModule mod = IRModule());

/*!
 * \brief Partition all matches of a DFPattern inside an Expr into separate Function calls
 *
 * \param pattern The pattern to match
 * \param expr The expression to patition
 * \param attrs A set of parameter names and values to apply to the partitioned function
 * \param check A callback function for checking more complicated properties of the matched
 * expressions, returns true if the match is accepted and false otherwise
 *
 * \return Return the paritioned Expr.
 */
Expr MNMPartitionPattern(DFPattern pattern, Expr expr, Map<String, ObjectRef> attrs,
                         tvm::PackedFunc check);

}  // namespace ir
}  // namespace mnm
