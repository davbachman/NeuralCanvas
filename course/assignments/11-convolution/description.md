<!-- Generated from description.html; run course/scripts/sync-handouts.py after editing. -->

# Assignment 11: Share features across an image

A fully connected layer gives each image location its own weights. A convolution instead reuses a small filter across locations. Build both models in [Neural Canvas](https://davbachman.github.io/NeuralCanvas/) and compare their predictions and parameter requirements.

**Submit:** **mlp.json** and **cnn.json**, worth 50 points each. Build both models yourself from a blank canvas; do not import the app's trained image example.

## Choose the data

Select **Dataset → Details → Browse datasets…**, choose **Images → Handwritten digits**, and click **Use this dataset**. Keep the default split and **Output mode → One example**.

| Property | Value |
| --- | --- |
| Input | 8 × 8 × 1 image, pixels already normalized to 0–1 |
| Target | Digit class 0–9, shape \[1\] |
| Training / validation | 400 / 100 images |

These are UCI 8×8 handwritten digits, not MNIST. The app calls the held-out partition a test set; because you inspect it to compare models, it serves as validation here. Keep the original rows and split. Do not divide the normalized input by 16 again.

## Part A: An image MLP — 50 points

Flatten the image into one row with a Tensor transform set to **Reshape**. Build a 16-unit hidden layer with ReLU and a linear ten-logit output layer. Use separate matrix Params and bias Params for the two layers. Determine all Param shapes from the input, hidden width, and output size. Keep the output shaped as one row of ten logits.

Connect raw logits and the Dataset target to **Loss → Cross entropy (logits)**. Use no parameter penalty. You may add a Softmax branch for inspection, but feed logits directly to Loss. Train with the settings below and save **mlp.json**.

## Part B: A convolutional classifier — 50 points

Start a fresh model on the same dataset. Add a **Convolution** block with four 3×3 filters, each spanning the image's one channel. Connect a filter Param and a bias Param. This operation uses stride 1 and valid convolution: it adds no border padding.

Apply ReLU, then **Average pooling**, which averages each 2×2 patch with stride 2. Flatten the pooled result to one row and build a linear ten-logit classifier with a matrix and bias. Determine the intermediate shapes and the classifier's input width. Use the same cross-entropy Loss and target as in your MLP. Train and save **cnn.json**.

Inspect values and gradients on a filter Param. Each filter weight participates in many local computations, and its gradient accumulates those contributions. Compare the models' parameter counts and validation predictions. A CNN need not outperform the MLP on this small dataset; this construction demonstrates locality and parameter sharing.

## Initialization and training

Initialize each model's first weight tensor with **Xavier**, random seed **137**, and its final classifier matrix with Xavier, seed **139**. Initialize both bias Params with **Zeros**. Keep every Param trainable, with no constraint. Set shapes before initializing.

| Train control | Setting for both models |
| --- | --- |
| Training execution | Trace engine · SGD |
| Learning rate | 0.03 |
| Epochs per run | 30 |
| Report loss every | 5 |
| Examples per update | 1 |
| Reshuffle training examples each epoch | On |

Use the Trace engine for both constructions: the tensor engine does not currently support Convolution or Average pooling. Keep **Loss → Details → Penalty → None**. Run inference on training and held-out data after training. If training is interrupted, continue without randomizing; the epoch count itself is not graded. Optional further training may improve or worsen validation results.

## Grading

Each model earns 15 points for the required architecture, 15 for training cross-entropy at most **0.40**, and 20 for validation cross-entropy at most **0.60** with accuracy at least **85%**. Performance credit requires the assigned architecture and original data. Neither model must beat the other.

The grader reconstructs predictions from the saved graph and parameters on the canonical images; it ignores cached metrics and claimed training history. Models must have exactly four independent Params. Use at most 100 blocks, 200 wires, and 4,096 total parameter values. Allowed constructions are the specified reshape, matrix/bias, ReLU, convolution/pooling, logits and Loss paths, optional Softmax, and connected aliases. Do not add feature scaling, dropout, penalties, hidden layers, or extra filters.

Reopen both files with **File → Import** before submitting. No written responses, screenshots, or Python files are required.

Version: convolution-11-v1. Source: E. Alpaydin and C. Kaynak, Optical Recognition of Handwritten Digits, UCI, [DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49), CC BY 4.0. The app uses a deterministic educational split of 500 records from the distributed dataset; this is not the original writer-independent UCI split.
