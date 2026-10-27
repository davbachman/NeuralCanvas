<!-- Generated from description.html; run course/scripts/sync-handouts.py after editing. -->

# Assignment 13: Give the generator a longer memory

**Your starting point and handoff**

| Step | What to use |
| --- | --- |
| Start with | your saved **base.json** (the unchanged bigram model) |
| Modify | Keep the Dataset, word embeddings, loss and Softmax. Replace the final-token input path with a flattened context and hidden MLP; resize the output projection. |
| Save and keep | **context.json**; use a copy in the next assignment. |

The phrases “liam entered” and “now liam is in the” end with the same token. A bigram model must give them the same next-token distribution. Build a model that can use the **ordered 32-token context**.

Use the narrative configuration selected below in your Dataset block. It keeps the same football narratives, **25-token vocabulary**, **32-token input windows** and training/held-out split. There is one target after the final input token. Keep the supplied data and split unchanged. Held-out data is validation during development.

Select your Dataset block and open **Details → Browse datasets…**. Choose **Text and sequences** → **Football stories**. Choose the **Narrative generation** preset. Set **Training targets** to **Final position only**. Click **Use this dataset**. Keep the supplied data and split unchanged.

**Prepared dataset properties**

| Dataset property | Supplied value |
| --- | --- |
| Vocabulary entries | 25, including unknown and padding |
| Input token IDs | 32 |
| Position IDs | 32 |
| Next-token target IDs | 1 |
| Training windows | 1534 |
| Held-out windows | 765 |

## Make a change the bigram model cannot make

Start from your saved **base.json**, the unchanged trained bigram model. Use its original logits, loss and output Softmax; the extra sampling branches are not needed here. Remove the Slice that kept only the final token. Embed every token with width **16**, then preserve all vectors in order as one row. Feed that row through a learned matrix and bias, a **ReLU hidden layer with 64 units**, and a final learned matrix and bias producing one row of vocabulary logits.

Work out the flatten width and all affected Param shapes yourself. Retain the existing loss and output probability branch. This model uses order through its fixed input slots; do not add position embeddings. Randomize the completed architecture and train for the specified budget. Save **context.json**.

The grader checks that every input slot reaches the hidden layer. Its report includes parameter count as well as loss. The following assignment will use this cost as a point of comparison.

## Parameter-count hint

Your completed models should have the following counts. Count individual learned numbers, including embedding tables and biases, and count each shared Param only once.

- **context.json:** **34,857** trainable parameters.

## Train and save

First set **Training execution** to **Tensor engine · minibatches** to show the Tensor controls. Then follow the Train tab from top to bottom:

**Training settings — Train tab**

| Control | Value |
| --- | --- |
| Epochs per run | 5 |
| Report loss every | 1 |
| Examples per update | 32 |
| Reshuffle training examples each epoch | Checked |
| Exact learning rate | 0.003 |
| Training execution | Tensor engine · minibatches |
| Backend | Tensor CPU |
| Optimizer | Adam |
| Gradient norm limit (0 = off) | 1 |
| Early stopping patience (0 = off) | 0 |

**Loss block → Details → Parameter regularization**

| Control | Value |
| --- | --- |
| Penalty | None |

Tensor CPU is the validated backend. Auto or WebGL may be used; speed depends on the computer and browser. Use the same backend for timed comparisons.

**Param → Initialize choices**

| Param block | Initialize |
| --- | --- |
| Word embedding table | uniform |
| Other matrices | Xavier · matrices |
| Bias vectors | zeros |

Apply changed shapes, check the saved Initialize choices, then use **Train → Randomize parameters**. Remove unused Params after an architecture change. Each changed architecture starts fresh; evaluation copies retain their trained values.

## Optional: generate a continuation

In **Dataset → Details → Continue the text**, paste this prompt and try **Most likely** or **Sample → Generate 40 tokens**:

> nora entered bedroom . liam entered kitchen . nora took football . nora entered hall . nora dropped football . now the football is in the hall . end nora entered bedroom . liam entered kitchen . nora took football . nora dropped football . nora entered hall . now

The prompt is longer than 32 tokens, so the app uses its last 32. Short prompts need padding and may behave differently. The temperature and sampling controls introduced earlier apply to every generator in this sequence. Use the same prompt to compare Most likely with Sample at temperatures 0.5, 1 and 2, keeping Top-k at 25. Sampling changes token selection without training. A fluent continuation can still misstate a person’s location, the football’s location or who holds it. This inspection is optional; the submitted model itself is the graded artifact.

## Submit and grading

Points below are dataset / model form / training loss / held-out loss. Loss limits are upper bounds. Performance credit requires the assigned architecture. The grader recomputes results from saved parameters; cached scores and declared epoch counts are ignored.

| File | Points | Training loss ≤ | Held-out loss ≤ |
| --- | --- | --- | --- |
| context.json | 10 / 50 / 20 / 20 | 0.6 | 0.7 |

Accuracy, parameter counts, common final-position loss and generated continuations are informational. No required written answers, screenshots or unsubmitted investigations.

Assignment 13-context.
