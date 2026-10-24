# Included teaching datasets

These are data files, not starter models or assignment solutions. The dataset
library loads only the selected file, and saves its data in the normal editable
project. No account or source download is needed after the dataset has been added.

The original classroom files retain their exact rows, vocabulary/label mappings,
preprocessing settings and fixed splits. They are shared across assignments when
identical. Their SHA-256 hashes are recorded in
[src/learning/dataset-library.json](../../src/learning/dataset-library.json).

| Family | Files and preparation | Source |
| --- | --- | --- |
| Auto MPG | mpg.csv; mpg-regularization.csv has 24 training rows and the same 80 held-out rows. Weight, model year, fuel economy. | [UCI Auto MPG](https://doi.org/10.24432/C5859H), CC BY 4.0 |
| Breast cancer | breast-cancer.csv: mean radius, mean texture, and malignant label. | [UCI Wisconsin Diagnostic Breast Cancer](https://doi.org/10.24432/C5DW2B), CC BY 4.0; distributed by scikit-learn |
| Iris | iris.csv: four measurements, three species, stratified fixed split. | R. A. Fisher, [UCI Iris](https://doi.org/10.24432/C56C76), CC BY 4.0; distributed by scikit-learn |
| Concrete | concrete.csv: eight raw features; identical input vectors stay in one split. | I-Cheng Yeh, [UCI Concrete Compressive Strength](https://doi.org/10.24432/C5PK67), CC BY 4.0; [CSV mirror](https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/concrete.csv) |
| Rings | rings.csv: 900 synthetic two-dimensional points and three classes. | Generated for Neural Canvas |
| Movie reviews | imdb.json: balanced subset, 1,200 train/400 held out; training-fitted 1,000-entry vocabulary, up to 128 tokens per review. | Maas et al. (ACL 2011), [Stanford Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) |
| Football stories | football-counts.json, football-tokens.json, stories.json, football-test.json, football-narratives-last.json, football-narratives-all.json, story-questions.json. | Synthetic five-event histories generated for Neural Canvas |
| Wine | wine.csv: 178 rows, 13 original numeric features, cultivar IDs 0–2. | [UCI Wine](https://doi.org/10.24432/C5PC7J), CC BY 4.0; distributed by scikit-learn |
| Diabetes | diabetes.csv: 442 rows, 10 raw baseline features, one-year progression target. | Efron et al., [scikit-learn diabetes data](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) |
| Palmer penguins | penguins.csv: 342 complete numeric records; two missing-measurement rows omitted; four body measurements and species. | [palmerpenguins](https://allisonhorst.github.io/palmerpenguins/), Horst, Hill and Gorman, CC0; Gorman, Williams and Fraser (2014) |
| California housing | california-housing.csv: 20,433 complete records; 207 missing-bedroom rows omitted; eight raw numeric inputs and median house value in dollars; ocean proximity excluded. | 1990 California census / StatLib; [Hands-On Machine Learning CSV mirror](https://github.com/ageron/handson-ml2/tree/master/datasets/housing) |

The four additional tabular datasets use deterministic 80/20 splits (stratified
for classification), prepared with seed 20261024. Original row order is retained.
`python scripts/prepare-library-datasets.py` reproduces these four files from
vendored source bytes in scripts/dataset-sources. Source hashes are recorded in
scripts/dataset-source-sha256.json. No network access or third-party Python
packages are needed to reproduce them.

The existing synthetic examples and UCI 8×8 handwritten digits remain in their
original app catalog; the browser includes them without duplicating their data.
See docs/CNN-DATA.md for digit attribution and preparation.

## Football configurations

- Small football-location classification: identical 480/240 examples with word
  counts (16 vocabulary entries) or ordered token IDs (17 entries).
- Mixed-question classification: 3,840/1,920 examples, separate story/question
  inputs and padding masks, 25 vocabulary entries and 9 answer classes.
- Football benchmark: the same 3,840 mixed training questions; 240 held-out
  football-location questions. Those held-out cases match the smaller set.
- Narrative generation: no questions. Same text and split in both files;
  32-token windows with either one final-position target or 32 shifted targets.
  There are 1,534 training and 765 held-out windows, from 960/480 histories.
- Answer-token prediction: identical question/answer records and splits to the
  football benchmark; combined 23-token input, with the answer excluded and used
  only as the target.

Mixed questions ask where the football, Liam, and Nora are, and who has the
football. Narrative statements cover the same locations and possession state.
Related history variants stay in one split. Vocabulary is fitted on training text
with declared reserved tokens. If held-out data is used for model or checkpoint
selection, it is validation data rather than an untouched final test.
