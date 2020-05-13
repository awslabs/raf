## RFCs in MNM
MNM adopts the model that RFC is required for every important design decision.

**What is an RFC.** Request for Comments, or RFC, is a stage in open source community, in which developers propose ideas and request feedbacks from the community.

**Why RFC.** Writing RFCs, although adds to the overhead of documentation writing, enjoys the benefit of visibility, trust earning, early correction for potential mistakes and future memorization.

***

## Levels of RFC Readiness

Inspired by the data readiness levels in machine learning [1], RFCs in MNM are categorized by levels of readiness in the similar way. An RFC is "accepted" means that it comes at least A3, "closed" means it is in level D or S; "work in progress" means A0-A2; "stabilized" means B1 or B2. Note that it is not necessary for an RFC to stabilize, unless it fundamentally changes the system, or adds a significant functionality that may affect future users or developers.
- D: Deprecated, which means that agreement does not happen for now.
- S: Superseded, which means there are better ideas proposed and the community agree to close the current one.
- A0: On plan, no formal ideas or implementation.
- A1: Formal ideas proposed.
- A2: Formal implementation proposed.
- A3: Implementation is merged into the master branch.
- B1: Refactor has been done to stabilize the implementation of the RFC.
- B2: The RFC is formally documented into developer's guide.

***

## List of RFCs

Below is the list of RFCs that are currently available.


|                                                              | Title                                               | Readiness Level |
|--------------------------------------------------------------|-----------------------------------------------------|-----------------|
| [[RFC-001]](https://github.com/meta-project/meta/issues/13)  | Unifying imperative and symbolic with NDArray       | B1              |
| [[RFC-002]](https://github.com/meta-project/meta/issues/20)  | Kernels and operators                               | S               |
| [[RFC-003]](https://github.com/meta-project/meta/issues/21)  | Data movement for heterogeneous/distributed runtime | D               |
| [[RFC-004]](https://github.com/meta-project/meta/issues/30)  | Operator interface                                  | A3              |
| [[RFC-005]](https://github.com/meta-project/meta/issues/41)  | Stream                                              | A3              |
| [[RFC-006]](https://github.com/meta-project/meta/issues/118) | Integrating CUDNN to MNM                            | B1              |
| [[RFC-007]](https://github.com/meta-project/meta/issues/90)  | Pure C Operator Interface                           | A2              |
| [[RFC-008]](https://github.com/meta-project/meta/issues/91)  | Configurable Gluon w/ Full Experiment Pipeline      | A0              |
| [[RFC-009]](https://github.com/meta-project/meta/issues/95)  | Refactor Operator Interface                         | B1              |
| [[RFC-010]](https://github.com/meta-project/meta/issues/108) | Introducing Reverse Mode Auto Diff (Symbolic)       | A3              |
| [[RFC-011]](https://github.com/meta-project/meta/issues/110) | Model                                               | A3              |
| [[RFC-012]](https://github.com/meta-project/meta/issues/124) | Imperative Mode Auto-Diff                           | A0              |
| [[RFC-013]](https://github.com/meta-project/meta/issues/128) | JITing Operators from TOPI                          | A2              |


***


## References
[1] Lawrence, Neil D. "Data readiness levels." arXiv preprint arXiv:1705.02245 (2017).
