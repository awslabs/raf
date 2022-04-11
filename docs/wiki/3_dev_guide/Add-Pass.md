<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Add a Compiler Pass

A compiler pass is used to analyze or transform RAF IR. As RAF IR is extended from Relay, most of RAF passes use the pass infrastructure of Relay. Please see TVM's dev guide firstly: [Adding a Compiler Pass to Relay](https://tvm.apache.org/docs/dev/how_to/relay_add_pass.html).

In the following sections, this article will introduce the process of how to add a pass to RAF firstly. Then there will be the differences you should know between RAF and TVM pass, as well as how to avoid stack overflow.

## Process

1. put your code under `src/pass`, e.g. pass `FoldConstant` is in `src/pass/fold_const.cc`
2. register the pass, similar with TVM but use RAF's API (`CreateRAFFunctionPass` and `RAF_REGISTER_GLOBAL`):
```c++
Pass FoldConstant() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return Downcast<Function>(fold_const::ConstantFolder().Mutate(f));
  };
  return CreateRAFFunctionPass(pass_func, 1, "FoldConstant", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.FoldConstant").set_body_typed(FoldConstant);
```
3. also add pass declaration `Pass FoldConstant();` to `include/raf/pass.h`, so that the pass can be used in other source code files
4. run codegen with `scripts/src_codegen/run_all.sh`, you will see the auto-generated FFI in python, the you can use the pass in python

## Differences

### Process Constant Node

Relay's pass infra traverses `RelayConstantNode`, but sometimes we need to process on RAF `ConstantNode` which has `raf::Value` in, we should get it by casting at first:

```c++
void VisitExpr_(const RelayConstantNode* op) override {
  const ConstantNode* node = static_cast<const ConstantNode*>(op);
  // do something
}
```

### Create Var Node

RAF uses `ExtendedVar` instead of Relay's `Var`, whenever we need a new var, use `MakeVar`.

For example, create a new var with empty type annotation:

```c++
Var new_var = MakeVar("name_of_var", {});
```

## Avoid Stack Overflow

This part mainly introduces how to avoid stack overflow for ANF IR. For GNF IR, just use `MixedModeVisitor/Mutator` instead of `ExprVisitor/Mutator`, or refactor the pass by using `ExprRewriter` and `PostOrderRewrite`.

When processing `LetNode`, use a loop or the utility function `ExpandANormalForm`. Choose the approach you think is best for code readability.

### Examples of using a loop:

For visitor:

```c++
void VisitExpr_(const LetNode* ln) final {
  Expr expr = GetRef<Let>(ln);
  // Iteratively visit let nodes to avoid stack overflow.
  while (expr->IsInstance<LetNode>()) {
    Let let = Downcast<Let>(expr);
    // do something
    expr = let->body;
  }
  // Visit the last body
  MixedModeVisitor::VisitExpr(expr);
}
```

For mutator:

```c++
Expr VisitExpr_(const LetNode* node) {
  scopes_.emplace_back(new LetList);
  auto scope = scopes_.back().get();
  Expr body;
  do {
    // do something, then push to scope
    scope->Push(new_var, new_value);
    body = node->body;
    node = body.as<LetNode>();
  } while (node);
  auto new_body = VisitExpr(body);
  auto ret = scopes_.back()->Get(new_body);
  scopes_.pop_back();
  return ret;
}
```

### Examples of using `ExpandANormalForm`:

In regular usage, it usually needs memory/cache/counter to avoid nested function call: when the first time LetNode’s body is visited in post_visit, it visits the var node (first time `post_visit` is called) and update the counter/cache, so in the following `post_visit` calls, it actually accesses the counter/cache without visiting again.

For ExprVisitor/MixedModeVisitor, the default implementation should be:

```c++
void VisitExpr_(const LetNode* op) final {
  auto pre_visit = [this](const LetNode* op) {
    this->VisitExpr(op->var);
    this->VisitExpr(op->value);
  };
  auto post_visit = [this](const LetNode* op) {
    this->VisitExpr(op->body);
    this->visit_counter_[op] += 1;  // avoid call nestedly
  };
  ExpandANormalForm(op, pre_visit, post_visit);
}
```

It is the same as:

```c++
void VisitExpr_(const LetNode* op) {
  this->VisitExpr(op->value);
  this->VisitExpr(op->var);
  this->VisitExpr(op->body);
}
```

For ExprMutator/MixedModeMutator, the default implementation should be:

```c++
Expr VisitExpr_(const LetNode* op) {
  auto pre_visit = [this](const LetNode* op) {
    // Rely on the Memoizer to cache pre-visit values
    this->VisitExpr(op->var);
    this->VisitExpr(op->value);
  };
  auto post_visit = [this](const LetNode* op) {
    // Rely on the Memoizer to cache pre-visit values
    Var var = Downcast<Var>(this->VisitExpr(op->var));
    Expr value = this->VisitExpr(op->value);
    // Visit body and cache the op
    Expr body = this->VisitExpr(op->body);
    auto expr = GetRef<Expr>(op);
    if (var.same_as(op->var) && value.same_as(op->value) && body.same_as(op->body)) {
      this->memo_[expr] = expr;  // avoid call nestedly
    } else {
      this->memo_[expr] = Let(var, value, body);  // avoid call nestedly
    }
  };
  ExpandANormalForm(op, pre_visit, post_visit);
  return memo_[GetRef<Expr>(op)];
}
```

It is the same as:

```c++
Expr VisitExpr_(const LetNode* op) {
  Var var = Downcast<Var>(this->Mutate(op->var));
  auto value = this->Mutate(op->value);
  auto body = this->Mutate(op->body);

  if (var.same_as(op->var) && value.same_as(op->value) && body.same_as(op->body)) {
    return GetRef<Expr>(op);
  } else {
    return Let(var, value, body, op->span);
  }
}
```

If the var/value has been mutated in pre_visit, when we use Mutate/VisitExpr, we will get it from cache (memo_).

### To refactor a function that visits let node using ExpandANormalForm:

1. if not override, copy the default implementation mentioned above. otherwise,
2. find code segment that visits `op->body`: `VisitExpr(op->body)`/`Mutate(op->body)`
3. copy the logic before that to `pre_visit`, after that to `post_visit`
4. in `post_visit`, if a visitor, add 1 to the counter
5. in `post_visit`, if a mutator, put the return value to cache

**Examples(see https://github.com/awslabs/raf/commit/1dbc22b904a26d9bc0a5306c2d4a0c70530cbc4c):**

1. a simple example, note that `ExprVisitor::VisitExpr_(op)` is equal to visit `op->var`+`op->value`+`op->body`

```c++
// before:
void VisitExpr_(const LetNode* op) final {
  this->Update(op->var, nullptr, kOpaque);
  this->Update(op->value, nullptr, kOpaque);
  this->Update(op->body, nullptr, kOpaque);
  let_binding_.emplace(op->var, op->value);
  ExprVisitor::VisitExpr_(op);
}

// after:
void VisitExpr_(const LetNode* op) final {
  auto pre_visit = [this](const LetNode* op) {
    this->Update(op->var, nullptr, kOpaque);
    this->Update(op->value, nullptr, kOpaque);
    this->Update(op->body, nullptr, kOpaque);
    let_binding_.emplace(op->var, op->value);
    this->VisitExpr(op->var);
    this->VisitExpr(op->value);
  };
  auto post_visit = [this](const LetNode* op) {
    this->VisitExpr(op->body);
    this->visit_counter_[op] += 1;
  };
  ExpandANormalForm(op, pre_visit, post_visit);
}
```

2. because body is visited in an if structure, we keep if-else logic in both pre_visit and post_visit

```c++
// before
Expr VisitExpr_(const LetNode* op) override {
  Expr ovalue = op->value;
  Var var = op->var;
  Expr value = VisitExpr(ovalue);
  if (value.as<ConstantNode>()) {
    memo_[var] = value;
    return VisitExpr(op->body);  // visit body
  }
  const VarNode* v = value.as<VarNode>();
  if (v && var_value_map_.count(v)) {
    var_value_map_[op->var.get()] = var_value_map_[v];
  } else {
    var_value_map_[op->var.get()] = value;
  }
  var->checked_type_ = value->checked_type();
  Expr body = VisitExpr(op->body);  // visit body
  Let let(var, value, body);
  let->checked_type_ = body->checked_type();
  return let;
}

// after:
Expr VisitExpr_(const LetNode* op) override {
  auto pre_visit = [this](const LetNode* op) {
    Expr ovalue = op->value;
    Var var = op->var;
    Expr value = VisitExpr(ovalue);
    if (value.as<ConstantNode>()) {
      memo_[var] = value;
    } else {
      const VarNode* v = value.as<VarNode>();
      if (v && var_value_map_.count(v)) {
        var_value_map_[op->var.get()] = var_value_map_[v];
      } else {
        var_value_map_[op->var.get()] = value;
      }
      var->checked_type_ = value->checked_type();
    }
  };
  auto post_visit = [this](const LetNode* op) {
    auto expr = GetRef<Expr>(op);
    Expr value = this->VisitExpr(op->value);  // get the cached value
    Expr body = this->VisitExpr(op->body);
    if (value.as<ConstantNode>()) {
      this->memo_[expr] = body;
    } else {
      Let let(op->var, value, body);
      let->checked_type_ = body->checked_type();
      this->memo_[expr] = let;
    }
  };
  ExpandANormalForm(op, pre_visit, post_visit);
  return memo_[GetRef<Expr>(op)];
}
```

3. use map to store local variable(s) (`alias` here) that used in both `pre_visit` and `post_visit`

```c++
// before:
Expr VisitExpr_(const LetNode* let) {
  if (let->value->IsInstance<TupleNode>()) {
    tuple_map_.emplace(let->var, Downcast<Tuple>(let->value));
  }
  auto new_value = VisitExpr(let->value);
  bool alias = false;
  if (new_value->IsInstance<VarNode>()) {
    auto alias_var = Downcast<Var>(new_value);
    alias_map_.emplace(let->var.get(), alias_var);
    alias = true;
  }
  auto new_body = VisitExpr(let->body);
  if (alias) {
    return new_body;
  }
  return Let(let->var, new_value, new_body);
}

// after:
Expr VisitExpr_(const LetNode* let) {
  std::unordered_map<Expr, bool, ObjectPtrHash, ObjectPtrEqual> let_alias_map;
  auto pre_visit = [this, &let_alias_map](const LetNode* op) {
    Expr expr = GetRef<Expr>(op);
    if (op->value->IsInstance<TupleNode>()) {
      tuple_map_.emplace(op->var, Downcast<Tuple>(op->value));
    }
    auto new_value = this->VisitExpr(op->value);
    let_alias_map[expr] = false;
    if (new_value->IsInstance<VarNode>()) {
      auto alias_var = Downcast<Var>(new_value);
      alias_map_.emplace(op->var.get(), alias_var);
      let_alias_map[expr] = true;
    }
  };
  auto post_visit = [this, &let_alias_map](const LetNode* op) {
    auto expr = GetRef<Expr>(op);
    auto new_body = VisitExpr(op->body);
    if (let_alias_map[expr]) {
      this->memo_[expr] = new_body;
    } else {
      auto new_value = this->VisitExpr(op->value);
      this->memo_[expr] = Let(op->var, new_value, new_body);
    }
  };
  ExpandANormalForm(let, pre_visit, post_visit);
  return memo_[GetRef<Expr>(let)];
}
```
