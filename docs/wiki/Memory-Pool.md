# Memory Pool

This document introduces the Memory Pool of Meta.

## Strategies

Currently, there are two types of memory pool in meta: (1) no_pool, (2) page_unit_pool.
By default, we choose page_unit_pool as our memory pool, which could bring down the running time by almost 50% for rn50/vgg/etc compared with no_pool. 

The memory usage of these two strategies are similar. Here is an experiment on ResNet50 with Tesla T4 (15109MB)
BatchSize | NoPool/MB | PageUnitPoo/MB
| :---------: | :----------: | :----------------: |
| 32 | 11160 | 11280 |
| 64 | 13018 | 13134 |
| 80 | 14032 | 14114 |
| 100 | 14728 | out of memory |

We can see that the page_unit_pool has slightly more memory cost compared with no_pool. What's more, the additional memory cost of page_unit_pool is requested in the first iteration(The iteration which will compile the ops).

## Change strategy

The default memory pool strategy is page_unit_pool. If you want to use no_pool, you can change it through Python API `InitPool(context, pool_name)`. Here is an example:

``` python
# Using default strategy
import mnm

device = 'cuda'

data = ...
label = ...
model = ...
mode.to(device)
loss = model(data, label)
loss.backward()
...

```

``` python
# Changing to no_pool strategy
import mnm
from mnm._ffi.memory_pool import InitPool, RemovePool
from mnm._core.core_utils import str2ctx

device = 'cuda'
pool_name = 'no_pool'
InitPool(str2ctx(device), pool_name)

data = ...
label = ...
model = ...
mode.to(device)
model.train_mode()
loss = model(data, label)
loss.backward()
...

```

if you want to change back to default memorpy strategy (page_unit_pool), you can call `RemovePool(contex)` or `InitPool(contxt, "page_unit_pool")`. (Note: Everytime you call InitPool, the current pool will be removed first, even if the new pool's name is equal to the current one.) If change you memory pool in the middle of your code, the ndarray will not be freed until there is no reference to it.

``` python
# Changing back to default strategy
import mnm
from mnm._ffi.memory_pool import InitPool, RemovePool
from mnm._core.core_utils import str2ctx

device = 'cuda'
# Start with no_pool
InitPool(str2ctx(device), 'no_pool')

data = ...
label = ...
model = ...
mode.to(device)
model.train_mode()
for i in range(10):
    if i%2 == 0:
        # Use no_pool for the even iter
        InitPool(str2ctx(device), 'no_pool')
    else:
        # Use page_unit_pool for the odd iter
        InitPool(str2ctx(device), 'page_unit_pool')
    loss = model(data, label)
    loss.backward()
model.infer_mode()
# Use the default pool strategy
RemovePool(str2ctx(device))
model(m_x)
...

```

## Design a new memory pool

Maybe you will want to develop your own memory pool, if so, you can follow the following instructions.

### Step1 Prepare

You should create a new folder under $META_HOME/src/memory_pool, and create a new cpp file that named as same to the folder name (recommended). The recommended name should be like `xxx_pool`.

### Step2 Implement your pool

To begin, you need include `"mnm/device_api.h"`,`"mnm/memory_pool.h"`, `"mnm/registry.h"`, and wrapper your code with namespace `mnm->memory_pool->your_pool`.
You will first need a memory wrapper that holds the actual memory. It must derived from `mnm::memory_pool::Memory`.

Then you can create the Pool Class that derived from `mnm::memory_pool::MemoryPool`.

### Step3 Register your pool

Remember to register your pool in the cpp file you created, the code should be like:
`MNM_REGISTER_GLOBAL("mnm.memory_pool._make.your_pool").set_body_typed(YourPool::make);`

After re-make meta, you can enable your pool by calling `InitPool(contxt, pool_name)`.
