<!-- Generated from description.html; run course/scripts/sync-handouts.py after editing. -->

# Assignment 17: Let attention build on attention

**Your starting point and handoff**

| Step | What to use |
| --- | --- |
| Start with | Your saved **all\_positions.json** (Part B of the previous assignment) |
| Modify | Save the trained model as one\_block.json. In a separate copy, duplicate and reconnect the transformer block with independent Params, then randomize the two-block model. |
| Save and keep | **one\_block.json** and **stacked.json**; keep stacked.json for the next assignment. |

In a single block, Q, K and V come from the initial token representations. In a second block, they can use representations that already contain information gathered by the first block. This is additional depth.

Use the narrative configuration selected below in your Dataset block. It keeps the same football narratives, **25-token vocabulary**, **32-token input windows** and training/held-out split. The targets contain one next-token ID for each input position. Keep the supplied data and split unchanged. Held-out data is validation during development.

Continue using model width **16** and MLP hidden width **32** unless a step explicitly changes a head’s width.

Select your Dataset block and open **Details → Browse datasets…**. Choose **Text and sequences** → **Football stories**. Choose the **Narrative generation** preset. Set **Training targets** to **Every input position**. Click **Use this dataset**. Keep the supplied data and split unchanged.

**Prepared dataset properties**

| Dataset property | Supplied value |
| --- | --- |
| Vocabulary entries | 25, including unknown and padding |
| Input token IDs | 32 |
| Position IDs | 32 |
| Next-token target IDs | 32 |
| Training windows | 1534 |
| Held-out windows | 765 |

## Use contextual information a second time

Keep the trained **all\_positions.json** from the previous assignment, Part B, as **one\_block.json**. In a separate copy, construct a second complete transformer block after the first block’s final residual addition and before the final Layer norm and vocabulary projection. You may select the first block’s components and use **Edit → Duplicate**, then reconnect the copy.

The second block needs its **own** Q, K, V and output matrices, two MLP matrices and biases, two Layer norms with independent scales and offsets, and causal mask. Its two shortcuts carry the second block’s input and its intermediate residual. Keep width **16**, MLP width **32**, and one head per block. Embed tokens and positions once; normalize and project to vocabulary once at the end.

Figure out the connections independently. Randomize the completed two-block model and train from fresh parameters. Submit it as **stacked.json** alongside **one\_block.json**. Do not reuse the first block’s Params in the second. Extra depth adds cost and compositional capacity; it need not improve every small-dataset run.

## Parameter-count hint

Your completed models should have the following counts. Count individual learned numbers, including embedding tables and biases, and count each shared Param only once.

- **one\_block.json:** **3,529** trainable parameters.
- **stacked.json:** **5,689** trainable parameters.

## Train and save

First set **Training execution** to **Tensor engine · minibatches** to show the Tensor controls. Then follow the Train tab from top to bottom:

**Training settings — Train tab**

| Control | Value |
| --- | --- |
| Epochs per run | 10 |
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
| Word and position tables | uniform |
| Other matrices | Xavier · matrices |
| Bias vectors | zeros |
| Layer norm scales | ones |
| Layer norm offsets | zeros |

Apply changed shapes, check the saved Initialize choices, then use **Train → Randomize parameters**. Remove unused Params after an architecture change. Each changed architecture starts fresh; evaluation copies retain their trained values.

## Optional: generate a continuation

In **Dataset → Details → Continue the text**, paste this prompt and try **Most likely** or **Sample → Generate 40 tokens**:

> nora entered bedroom . liam entered kitchen . nora took football . nora entered hall . nora dropped football . now the football is in the hall . end nora entered bedroom . liam entered kitchen . nora took football . nora dropped football . nora entered hall . now

The prompt is longer than 32 tokens, so the app uses its last 32. Short prompts need padding and may behave differently. The temperature and sampling controls introduced earlier apply to every generator in this sequence. Use the same prompt to compare Most likely with Sample at temperatures 0.5, 1 and 2, keeping Top-k at 25. Sampling changes token selection without training. A fluent continuation can still misstate a person’s location, the football’s location or who holds it. This inspection is optional; the submitted model itself is the graded artifact.

## Submit and grading

Points below are dataset / model form / training loss / held-out loss. Loss limits are upper bounds. Performance credit requires the assigned architecture. The grader recomputes results from saved parameters; cached scores and declared epoch counts are ignored.

| File | Points | Training loss ≤ | Held-out loss ≤ |
| --- | --- | --- | --- |
| one\_block.json | 5 / 15 / 5 / 5 | 0.65 | 0.7 |
| stacked.json | 5 / 35 / 15 / 15 | 0.65 | 0.7 |

Accuracy, parameter counts, common final-position loss and generated continuations are informational. No required written answers, screenshots or unsubmitted investigations.

Assignment 17-stacking.
