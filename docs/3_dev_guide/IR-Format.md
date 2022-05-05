<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Intermediate Representation (IR) in RAF

Intermediate representation (IR) is the key to describe a program in a compiler. RAF extends the data structures from Relay in [Apache TVM](https://tvm.apache.org/) to represent deep learning workloads.

Despite the data structures, RAF has three major IR formats in use, namely A-Normal Form (ANF), Graph Normal Form (GNF) and Basic Block Normal Form (BBNF). In this document we give a brief introduction of each format, and discuss how we trade-off between the IRs for certain tasks. 

## A-Normal Form

A-Normal Form (ANF) was first introduced as a simplified alternative to continuation-passing style (CPS) in functional languages. Here is an example of ANF,

```
def @ANormalF(%x, %y) {
	let %f = fn(%a, %b) {
		let %v = add(%a, %b)
		%v
	}
	let %v1 = mul(%x, %x)
	let %v2 = %f(%v1, %y)
	let %v3 = div(%x, %y)
	let %v4 = %f(%v2, %v3)
	%v4
}
```

In ANF, each expression must be binded to a variable via `Let` binding. The `Let` node in RAF is defined as `Let(var, value, body)`, in which `var` is the variable and `value` is the expression to bind. `body` specifies the next statements to evaluate, it is always either another `Let` or the return variable of the function (`%v` and `%v4` in the above example). Once evaluated, `var` can be used in the remaining statements, i.e., those in `body`.

We can get some high-level ideas without diving into PL theories.

Firstly, ANF explicitly defines the evaluation order. In the example above, `%v3` has to be evaluated after `%v2`, even though they don't have dependencies between each other.

Secondly, with expression being binded to a variable, it becomes easier to fetch it later via the binding in a pass. This is useful in some passes such as auto-differentiation, with everything being just variables and their adjoints during the propagation, complex expressions such as functions (`%f` in the above example) don't require special handling.


## Graph Normal Form

The downside of ANF is though, the difficulty to do pattern matching on a subgraph. This is when Graph Normal Form (GNF) comes into the picture, 

```
def @GraphF(%x, %mean, %var) {
	%1 = batch_norm(%x, running_mean=%mean, running_var=%var)
	%2 = equal(%mean, 2)
	if (%2) {
		mul(%1, %1)
	} else {
		%1
	}
}
```

In GNF, expressions are not connected via variable bindings. The IR is no longer linear as in ANF but is more like a graph that contains vertices and edges. One can traverse directly from an expression to another via an edge. In the example above, you can reach node `batch_norm` by visiting the first argument of `equal(%1, 2)`. Such graph structure is especially suitable for pattern matching and substitution. We find it is useful in many pass implementation.

However, GNF may result evaluation order ambiguity. This could be an issue when the IR contains side-effect. In the example above, `%1` technically resides in the if-else scope that can be evaluted just when either branch is reached. In addition, `%1 = batch_norm(%x, running_mean=%mean, running_var=%var)` and `%2 = equal(%mean, 2)` don't have dependencies between each other from the IR's perspective. However, as `batch_norm` is a side-effect operation which may update `%mean`, it does matter in this case which expression is evaluated first, while GNF fails to capture the dependency.


## Basic Block Normal Form

BBNF is a hybrid of ANF and GNF that merges the advantages from both sides. Borrowed from [the idea in Apache TVM community](https://discuss.tvm.apache.org/t/basic-block-normal-form), we also adopted Basic Block Normal Form (BBNF) which is a trade-off between ANF and GNF. In BBNF, the IR is grouped into multiple basic blocks, within each block it is GNF, while for values that are being used in multiple blocks, they have to be referred by a variable. For instance, the example in GNF will be rewritten as,

```
def @BBlockNF(%x, %mean, %var) {
	let %v1 = batch_norm(%x, running_mean=%mean, running_var=%var) // block 0
	%2 = equal(%mean, 2)                                           // block 0
	if (%2) {                                                      // block 0
		mul(%v1, %v1)                                               // block 1
	} else {
		%v1                                                         // block 2
	}
}
```

By having the let bindings, the IR above implicitly defines the dependency and the execution order. Meanwhile the graph format within each basic block makes pass implementation much easier compare to ANF.


## Design Choices

Everything is an argument. RAF has both ANF and BBNF in use and they can be easily converted to each other. (There is GNF too but in most cases it is just used as an intermediate status when converting to BBNF). Passes require easy traversal and pattern matching use BBNF, while those require strict execution order and variable bindings, such as rematerialization, auto-differentiation, etc., work on ANF. Some passes, e.g., type inference, etc., can work on either format.

Use `raf::pass::ToANormalForm()` and `raf::pass::ToBasicBlockNormalForm()` to convert the IR format between each other. Remember to call `raf::pass::ToGraphNormalForm()` first before converting to BBNF.
