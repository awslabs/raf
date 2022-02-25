<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

**Meta framework.** Because, first, it is not a single framework, but meta by nature. With TVM Relay frontend importers, we are able to absort any TensorFlow model, ONNX model, into our core intermediate representation, and mix it with each other - as a researcher, just imagine that you are able to define the tedious part of your network in Keras, like backbones in vision, and we allow you to focus on manipulating the awesome part of your neural network using our numpy-like API.

**Compiler native.** Second, it is said that it is boring to develop another one because all frontends are converging. True. While we converge to Numpy + Gluon/Keras, we are the first compiler-native framework, which is the key feature that makes us the most outstanding. We are the missing frontend of TVM/Relay stack for intelligent compiler. Once you implement your model using our framework, our JIT is in charge of fusing operators, finding the best data layout for training.

**Python-Relay Transpiler.** Third, we are capable of translating Python to the properly-designed intermediate representation, Relay, supporting arbitrary control flow (nested loop/if/continue/break/return), mutual function calls and wide range of types, including primitive types\* and container types (tuple, list, dict), and classes\*\* as well as back propogation through them. In the meantime, we are able to hook in almost all developer/user-defined functions to enrich our system via either true hybridization or packed function registration.

\* Integers, floating point numbers, bool, and potentially strings.

\*\* Technically, we support class through named tuple, which is slightly different.

