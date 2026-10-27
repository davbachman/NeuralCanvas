<!-- Generated from description.html; run course/scripts/sync-handouts.py after editing. -->

# Assignment 5: When layers become a neural network

Build two models in [Neural Canvas](https://davbachman.github.io/NeuralCanvas/) to predict concrete compressive strength from its ingredients and age. First build a linear baseline. Then add a hidden layer and a nonlinear activation to build a multilayer perceptron (MLP). Submit two saved model files; no written responses are required.

## Why a nonlinear activation?

Suppose you compute H = XW₁ + b₁, then predictions = HW₂ + b₂. Substitution gives X(W₁W₂) + (b₁W₂ + b₂): still one linear regression model with an intercept. Two matrix products alone do not make a more expressive model.

Instead, compute H = ReLU(XW₁ + b₁), where ReLU(t) = max(0, t) is applied to every entry. The hidden layer contains 16 learned intermediate features. Its activations depend on the input, allowing the prediction to change slope. The final output is a strength, so leave it unrestricted: no sigmoid or softmax at the output.

## Data and shared setup

Add a Dataset block.

Select your Dataset block and open **Details → Browse datasets…**. Choose **Regression** → **Concrete strength**. Click **Use this dataset**. Keep the supplied data and split unchanged.

The 1,030 measured mixtures have **828 training and 202 held-out validation examples**. Rows with identical input features are kept in the same split. Confirm **Regression** and that `strength` is the target. Strength is measured in megapascals (MPa). Do not change the data or split labels.

1. Set Dataset output mode to **Numeric batch** and Rows to **Training**.
2. Connect all eight feature outputs to **Concatenate**, with Axis `1` and eight inputs, in this order: **cement, slag, ash, water, superplastic, coarseagg, fineagg, age**. Ingredient quantities are in kg/m³; age is in days. Do not include strength among the features.
3. Connect the resulting matrix to one **Standardize features** block. Click **Fit on training rows** in Details. Confirm 828 observations and eight means/scales. Its output X has shape B-by-8, where B is the number of examples being evaluated. Keep these training-fitted statistics fixed for validation.
4. Connect the Dataset's **strength** output to **Tensor transform → Reshape**, with Shape `-1, 1`. This is the **target** column: it will connect to the second input of Loss. The inferred row count works for both splits. Do not standardize the target.

## Part A: Linear baseline — 40 points

1. Create a weight **Param W**. In Details choose **Xavier**, set Random seed to `137`, enter Shape `8, 1`, and press **Enter in the Shape field** to initialize its eight values. Connect X and W to the first and second inputs of **Matrix product**.
2. Create a separate **scalar bias Param b** initialized to zero. Add it to the product with **Arithmetic** addition, underlying expression `x1 + x2`. The prediction XW + b has shape B-by-1.
3. Connect the predictions directly to the first input of **Loss → Mean squared error**, and the reshaped strength target to the second. Choose **Penalty: None**.
4. Train using the settings below, inspect both splits, then use **File → Save** and name the file `linear.json`.

## Part B: One hidden layer with ReLU — 60 points

Keep Part A's saved file. In a separate project or a copy of it, reuse the Dataset, Concatenate, fitted Standardize, and target reshape. Replace the learned prediction path with the following two layers.

1. Create **W₁** with Shape `8, 16`, choosing Xavier and Random seed `137` before entering the shape and pressing Enter. Confirm 128 values. Compute XW₁ with Matrix product.
2. Create **b₁** with Shape `16` and the **zeros** initializer; press Enter in Shape. Add it to XW₁ using Arithmetic addition. The 16 biases broadcast across examples.
3. Connect this sum to **Activation → ReLU**. Its output H is B-by-16. This is the hidden layer: 16 intermediate values for each example, rather than the final answer.
4. Create **W₂** with Shape `16, 1`, Xavier initialization, and Random seed `139`; press Enter in Shape. Compute HW₂ with a second Matrix product.
5. Add a separate **scalar bias b₂** initialized to zero. Connect these B-by-1 predictions to MSE Loss and the reshaped strength target to its second input. Keep Penalty: None. Use exactly these four Params, with no extra learned layers.
6. Train, evaluate both splits, and save as `mlp.json`. Compare its held-out MSE with your linear baseline. After held-out inference, inspect the ReLU output in **Data** with **Inspection context** set to held-out inference. Browse the evaluated rows with the arrows beside **Examples**; compare the block's input and output using **Inspect signal**. Negative pre-activation values become zero. Eight original features do not have a simple two-dimensional model plot; use the Data tab, loss history, and inference results.

## Shape checklist

| Quantity | Linear | MLP |
| --- | --- | --- |
| Standardized X | B × 8 | B × 8 |
| First weight Param | 8 × 1 | 8 × 16 |
| First bias | Scalar | Vector of length 16 |
| Hidden activation H | — | B × 16 |
| Output weight / bias | — | 16 × 1 / scalar |
| Predictions and target | Both B × 1 | Both B × 1 |

## Training and evaluation

In **Train → Training execution**, change the default Tensor engine to **Trace engine · SGD**. Use no loss penalty. Use Xavier for weight matrices and zeros for biases; initializing every weight to zero prevents hidden units from learning distinct features.

| Model | Learning rate | Examples per update | Epochs | Report every |
| --- | --- | --- | --- | --- |
| Linear | 0.03 | 828 | 4,000 | 500 |
| MLP | 0.003 | 828 | 6,000 | 500 |

Use **Train → Run epochs**. These runs can take several minutes or longer depending on your browser and computer; keep the tab foregrounded. You can run consecutive chunks without resetting parameters. Reporting shows training and held-out loss at the chosen interval. To evaluate separately, use **Test → Training set** or **Held-out test set → Run inference**. The app calls it a test set; here we use it as validation to compare models. Do not fit preprocessing on it.

Loss is **MSE in MPa²**, not RMSE. Its square root gives RMSE in MPa. A smaller training error alone does not establish better generalization: compare validation error too. The MLP has more parameters and takes longer; nonlinear models need not improve every dataset or every training run.

## Grading and submission

Part A awards 5 points for file/data, 15 for model form/scaling, and 10 each for training and validation performance. Part B awards 5 for file/data, 25 for model form/scaling, and 15 each for training and validation performance. Performance credit requires the assigned architecture and training-fitted scaling. Cutoffs are inclusive:

| Model | Training MSE at most | Validation MSE at most |
| --- | --- | --- |
| Linear | 120 | 150 |
| MLP | 45 | 65 |

Submit only **linear.json** and **mlp.json**, downloaded using File → Save. Reopen them with File → Import to verify the saved models. Each part is graded independently. The grader recomputes predictions from the saved operations and parameters; cached losses, values, and epoch counts do not establish credit. Use at most 100 blocks, 200 wires, 512 values per Param, and 32 MiB per file. Renamed blocks, visual groups, and connected Input/Target aliases are accepted.

Version mlp-05-v1. Data: I-Cheng Yeh, Concrete Compressive Strength, UCI DOI 10.24432/C5PK67, CC BY 4.0. Source CSV mirror: stedy/Machine-Learning-with-R-datasets. All original observations are retained, with a fixed split prepared for this assignment.
