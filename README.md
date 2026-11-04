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


## Running the Prototype

Install the project using Python 3.12:

```bash
python -m pip install -e ".[dev]"
```

Generate the fixed synthetic dataset once:

```bash
python -c "from arc_revision.data.generator import save_dataset; save_dataset('data/generated/v1')"
```

Train the two experimental conditions:

```bash
python scripts/train.py --config configs/baseline.yaml
python scripts/train.py --config configs/revision.yaml
```

Evaluate the selected checkpoints on the held-out test split:

```bash
python -m scripts.evaluate --checkpoint runs/baseline/best.pt --output runs/baseline/test_metrics.json
python -m scripts.evaluate --checkpoint runs/revision/best.pt --output runs/revision/test_metrics.json
```

Both training conditions use the same saved dataset and model architecture. Evaluation reports final program accuracy, query-grid exact-match accuracy, and intermediate valid-hypothesis probability mass, with breakdowns by ambiguity and demonstration count.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development conventions and reproducibility requirements.
