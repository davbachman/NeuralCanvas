<!-- Generated from description.html; run course/scripts/sync-handouts.py after editing. -->

# Assignment 7: What can word counts tell us?

How can a numerical model work with text? In [Neural Canvas](https://davbachman.github.io/NeuralCanvas/), first build a movie-review classifier with guidance. Then adapt the same kind of model to answer questions about short stories. Submit both trained models; the second build is your independent challenge.

## Part A — Movie reviews: guided build (60 points)

The **Movie reviews** dataset has 1,200 training and 400 held-out reviews, balanced between positive and negative. Add a Dataset block, then select the following library dataset:

Select your Dataset block and open **Details → Browse datasets…**. Choose **Text and sequences** → **Movie reviews**. Click **Use this dataset**. Keep the supplied data and split unchanged.

**Prepared dataset properties**

| Dataset property | Supplied value |
| --- | --- |
| Vocabulary entries | 1000 |
| Text tokens used per example | Up to 128 |
| Word-count output shape | 1 × 1000 |
| Answer classes | 2: negative and positive |
| Training examples | 1200 |
| Held-out examples | 400 |

Keep the supplied split. In Dataset Details, **Vocabulary** shows each token's index. Vocabulary fitting uses training reviews only; unfamiliar tokens share index 0, `<unk>`. Punctuation counts as tokens. The model receives counts from the first 128 tokens of each review, or the entire review if shorter. Vocabulary selection uses the full training reviews.

The word-count output for one review has shape `1, 1000`. Entry j counts occurrences of token j. A **one-hot vector** represents one token with a 1 at its vocabulary index and zeros elsewhere; summing these vectors gives word counts.

Build **logit = XW + b**. Connect the Dataset's word-count output directly to the model. Use a Xavier-initialized Param matrix of shape `1000, 1` and a zero-initialized bias vector of shape `1`. No concatenation or standardization is needed.

Connect the logit directly to **Binary cross entropy (logits)** and connect **sentiment** to the target input. Keep a separate **Sigmoid** branch for the probability of a positive review. Negative is class 0 and positive is class 1. Do not put Sigmoid before this Loss block.

Train from fresh parameters using the settings below. Save as **reviews.json**.

## Part B — Football stories: independent build (40 points)

Save Part A, then start a new model. The **Football location · word counts** preset has 480 training and 240 held-out questions. Each story has five events involving Nora, Liam, and a football. A person can enter a room, take the football, or drop it. A person carrying the football takes it along when moving. A dropped football stays in that room. Each question asks **where is football**, with one of six rooms as the answer.

> liam entered garden . nora entered kitchen . nora took football . nora dropped football . nora entered bedroom . where is football<br>**Answer:** kitchen

Select your Dataset block and open **Details → Browse datasets…**. Choose **Text and sequences** → **Football stories**. Choose the **Football location · word counts** preset. Click **Use this dataset**. Keep the supplied data and split unchanged.

**Prepared dataset properties**

| Dataset property | Supplied value |
| --- | --- |
| Vocabulary entries | 16 |
| Text tokens used per example | 23 |
| Word-count output shape | 1 × 16 |
| Answer classes | 6 rooms |
| Training examples | 480 |
| Held-out examples | 240 |

Inspect the Dataset’s output dimensions before creating Params. The dataset supplies word counts from each complete story and question.

Build a **linear multiclass classifier**: **logits = XW + b**. Determine the matrix and bias-vector shapes yourself. Use **Xavier · matrices** for the matrix and **zeros** for the bias. Use the counts directly, with no hidden layer or engineered features. Choose the loss and probability branch using the tools from earlier assignments. Connect the Dataset's original **answer** as the target.

The class indices are **0 bathroom, 1 bedroom, 2 garden, 3 hall, 4 kitchen, 5 office**. Train to meet the loss targets below and save **stories.json**. Accuracy is reported but does not earn or lose points. Once you meet the loss targets, you do not need to keep training to chase accuracy.

If the loss falls short, check the fitted input width, output width, target connection, loss, and optimizer. You may adjust the learning rate or duration while retaining the assigned architecture and prepared dataset.

## Training settings

First set **Training execution** to **Tensor engine · minibatches** to show the Tensor controls. Then follow the Train tab from top to bottom:

**Training settings — Train tab**

| Control | Movie reviews | Football questions |
| --- | --- | --- |
| Epochs per run | 5 | 100 |
| Report loss every | 1 | 10 |
| Examples per update | 32 | 32 |
| Reshuffle training examples each epoch | Checked | Checked |
| Exact learning rate | 0.01 | 0.02 |
| Training execution | Tensor engine · minibatches | Tensor engine · minibatches |
| Backend | Tensor CPU | Tensor CPU |
| Optimizer | Adam | Adam |
| Gradient norm limit (0 = off) | 0 | 0 |
| Early stopping patience (0 = off) | 0 | 0 |

**Loss block → Details → Parameter regularization**

| Control | Value |
| --- | --- |
| Penalty | None |

**Param → Initialize choices**

| Param block | Initialize |
| --- | --- |
| Weight matrix | Xavier · matrices |
| Bias vector | zeros |

Apply changed shapes, check the saved Initialize choices, then use **Train → Randomize parameters**. Tensor CPU is the validated backend for these small models; speed varies by computer and browser.

The held-out split serves as validation while you adjust a model. More training is not automatically better for held-out predictions.

## Inspect results while building

**Test → Held-out test set → Run inference** reports accuracy and individual predictions. Select your probability block and open **Data**, with **Inspection context** set to held-out inference, to browse probabilities. Match the **Dataset row** numbers to the report. To read the corresponding text, use Dataset Details → **Rows: Held out** → **Current row**. Changing Current row clears the inference report; rerun inference to restore the probability table.

These views help you diagnose and compare your submitted models; there is no separate written investigation. For the stories, held-out Dataset rows **480 and 481** provide a useful pair to inspect if you are puzzled by the results.

<details>
<summary>Optional experiment — reverse a review's input tokens</summary>

This is optional and ungraded; no extra file is required. In your trained review model, select a dataset row and use **Train → Step through a lesson → Run forward**. Note the Loss block's current-example value and the Sigmoid output. In Dataset Details, open **Word-order experiment**, click **Reverse token order**, and run forward again without training or randomizing parameters. Compare those same values. This reverses the selected input tokens, not the entire review before truncation. Whole-dataset inference uses the original reviews and does not evaluate this experiment. Selecting a dataset row restores its original input.

</details>

## Submit — two model files, 100 points

Use **File → Save** and upload **reviews.json** and **stories.json** to Gradescope. Keep the separate probability branch in each. No screenshots, quiz, or written answers are required.

| File | Training loss | Held-out loss | Held-out accuracy |
| --- | --- | --- | --- |
| reviews.json | ≤ 0.30 | ≤ 0.60 | ≥ 70% |
| stories.json | ≤ 1.25 | ≤ 1.45 | Reported, not graded |

Reviews: 5 points for file/data, 25 for model form and probability branch, 10 for training loss, 10 for held-out loss, and 10 for accuracy. Stories: 5 for file/data, 20 for model form and probability branch, 10 for training loss, and 5 for held-out loss. Performance credit requires the assigned architecture. The grader recomputes predictions on the original data using saved parameters; cached scores and losses are ignored.

Assignment: counts-07. Reviews: [Stanford Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/), Maas et al. (ACL 2011), via a public CSV mirror; this course subset uses its own split, not the official benchmark. Football stories: course-generated; paired event orders and both actor assignments stay together in the split. The IMDb data is unchanged. The held-out questions are validation data used to develop the assignment.
