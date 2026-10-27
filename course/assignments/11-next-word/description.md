<!-- Generated from description.html; run course/scripts/sync-handouts.py after editing. -->

# Assignment 11: Predict the next word

**Your starting point and handoff**

| Step | What to use |
| --- | --- |
| Start with | A new graph; keep your saved **attention.json** and **football.json** |
| Modify | Begin next-token generation with a deliberately small bigram model, using familiar embedding, matrix/bias, loss and Softmax operations. |
| Save and keep | **bigram.json**; keep it for the next assignment. Keep the previous assignment’s football.json for the capstone. |

Until now, a model has predicted a label. Here the label is the *next word or punctuation token*. After choosing a token, we can append it and predict another. Start with a **bigram model**: it sees only the final token, even though the Dataset supplies a longer context.

Choose **Narrative generation** in the dataset browser with **Training targets: Final position only**, as described below. This is a next-token prediction task with no questions. The synthetic football stories use **liam**, **nora** and six rooms. Each story has five events followed by a consequence. The vocabulary has **25 tokens**, including punctuation and reserved tokens. The words **where** and **who** are reserved for the final question-answer task; they do not appear in narrative training. Each input contains **32 token IDs**; one next-token target follows the final input position. The dataset supplies vocabulary, windows and the training/held-out split. Embedding width is a model choice specified below, not an attribute of the dataset.

Example story: *liam entered garden . nora entered kitchen . nora took football . nora dropped football . nora entered bedroom . now the football is in the kitchen . end* Stories are joined into streams within each split. A window can span an *end* marker. Room-layout groups stay in one split, including all their actor and event-order variants. The same history split will be used for questions in the final QA capstone. Keep the supplied data and split unchanged; held-out data is validation during development.

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

This Dataset supplies one token sequence for next-word prediction. The earlier MLP and cross-attention assignments supplied separate story and question inputs; here the position IDs describe slots in the single context window.

## Turn classification into generation

1. Use the Dataset selected above. Its outputs are token IDs, position IDs and targets. Use a Tensor transform → **Slice** on token IDs: axis **0**, start **31**, end **32**. End is exclusive; the result contains one ID.
2. Add an **Embedding** block. Connect a learned Param table to its table input and the retained ID to its ID input. Use a **16-dimensional embedding**; determine the table shape from the vocabulary size.
3. Use one learned matrix and bias to turn that embedding into one row of vocabulary logits. Determine their shapes. Connect logits directly to **Cross entropy (logits)** and connect the Dataset targets to its target input. Keep a separate **Softmax** branch so you can inspect probabilities.

Use the familiar Param shape field and press Enter to initialize a tensor. Save the trained model as **bigram.json**. Position IDs are not needed by this model.

## Parameter-count hint

Your completed models should have the following counts. Count individual learned numbers, including embedding tables and biases, and count each shared Param only once.

- **bigram.json:** **825** trainable parameters.

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

In **Dataset → Details → Continue the text**, paste this prompt and try **Most likely → Generate 40 tokens**:

> nora entered bedroom . liam entered kitchen . nora took football . nora entered hall . nora dropped football . now the football is in the hall . end nora entered bedroom . liam entered kitchen . nora took football . nora dropped football . nora entered hall . now

The prompt is longer than 32 tokens, so the app uses its last 32. Short prompts need padding and may behave differently. A fluent continuation can still misstate a person’s location, the football’s location or who holds it. The next assignment will introduce probabilistic generation using this same trained model. This inspection is optional; the submitted model itself is the graded artifact.

## Submit and grading

Points below are dataset / model form / training loss / held-out loss. Loss limits are upper bounds. Performance credit requires the assigned architecture. The grader recomputes results from saved parameters; cached scores and declared epoch counts are ignored.

| File | Points | Training loss ≤ | Held-out loss ≤ |
| --- | --- | --- | --- |
| bigram.json | 10 / 50 / 20 / 20 | 1.15 | 1.2 |

Accuracy, parameter counts, common final-position loss and generated continuations are informational. No required written answers, screenshots or unsubmitted investigations.

Assignment 11-next-word.
