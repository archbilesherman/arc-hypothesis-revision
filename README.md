# Learning to Revise Task Hypotheses from ARC Demonstrations

**Investigating whether neural models can preserve plausible explanations under ambiguity and selectively revise them as new evidence arrives.**

## Research Question

In the [Abstraction and Reasoning Corpus (ARC)](https://github.com/fchollet/ARC-AGI), a model must infer a transformation from a few input-output grid examples and apply it to a new grid. Early demonstrations may support multiple explanations, while later evidence can eliminate some while preserving others.

**Does explicitly supervising hypothesis revision improve a model's reasoning compared with supervising only its final answer?**

Rather than attempting unrestricted ARC, we investigate this question through controlled synthetic tasks with executable, compositional hypotheses whose consistency with observed evidence can be evaluated exactly.

## Approach

### Hypothesis Representation

Hypotheses are represented as executable programs in a parameterized domain-specific language (DSL). Programs compose geometric and color operations, with an extensible representation intended to support additional reasoning primitives.

Each episode contains its own candidate hypothesis space, rather than selecting from a globally fixed set of transformations.

### Sequential Hypothesis Revision

The proposed neural architecture uses a convolutional neural network (CNN) to encode demonstrations and a gated recurrent unit (GRU) to maintain a persistent representation of the task as evidence accumulates.

The model will score candidate programs after each demonstration, producing a distribution over plausible hypotheses.

We compare two training objectives under equivalent architectures and computational budgets:

- **Final-only baseline:** Receives supervision on the generating hypothesis after the final demonstration.
- **Revision-aware model:** Additionally receives intermediate supervision on hypotheses consistent with the evidence observed so far.

Intermediate targets