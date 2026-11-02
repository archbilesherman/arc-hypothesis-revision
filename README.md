# Learning to Revise Task Hypotheses from ARC Demonstrations

**Investigating whether neural models can preserve plausible explanations under ambiguity and selectively revise them as new evidence arrives.**

## Research Question

In the [Abstraction and Reasoning Corpus (ARC)](https://github.com/fchollet/ARC-AGI), a model must infer a transformation from a few input-output grid examples and apply it to a new grid. Early demonstrations may support multiple explanations, while later ones can rule out some without invalidating the rest.

**Does explicitly supervising hypothesis revision improve a model's reasoning compared with supervising only its final answer?**

Rather than attempting unrestricted ARC, we investigate this question within a controlled space of executable transformations, where the consistency of each hypothesis with observed evidence can be measured exactly.

## Approach

A shared **convolutional neural network (CNN)** encodes the input and output grids of each demonstration. A **gated recurrent unit (GRU)** processes these representations sequentially, maintaining an internal task state. After each demonstration, a classification head predicts a probability distribution over candidate transformations.

We compare two training objectives using the **same architecture and computational budget**:

- **Final-only baseline:** Receives supervision only on the generating transformation after the final demonstration.
- **Revision-aware model:** Additionally receives supervision on which hypotheses remain consistent after each demonstration.

The revision-aware model is trained using **version spaces**, which represent the set of transformation hypotheses consistent with all demonstrations observed so far. When multiple hypotheses remain valid, the intermediate training target distributes probability equally among them. As new evidence arrives, contradicted hypotheses are eliminated while still-consistent explanations retain support.

This allows us to study whether explicit supervision encourages the model to maintain uncertainty, incorporate new evidence, and selectively revise its internal representation rather than prematurely committing to a single explanation.

## Research Direction

We begin with synthetic ARC-style tasks and a controlled library of geometric transformations, constructing sequences in which individual demonstrations may be ambiguous but their combined evidence identifies a unique rule.

Our evaluation considers both **final prediction accuracy** and **intermediate revision behavior**, including how effectively models reduce confidence in contradicted hypotheses while preserving support for valid explanations.

Further experiments will explore compositional transformations, sensitivity to demonstration order, and ablations that distinguish genuine hypothesis revision from improvements attributable to additional supervision.

The broader objective is to investigate how neural systems can maintain and update structured interpretations of a problem as evidence accumulates.
