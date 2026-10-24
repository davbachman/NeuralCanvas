import assets from '../learning/dataset-library.json'
import { analyzeCustomCsv, parseCustomCsv } from './customCsv'
import { DATASET_OPTIONS, datasetExamplesForNode, datasetForNode } from './datasets'
import { decodeTokens, isTextDatasetData } from './textData'
import type { CustomCsvData, DatasetKind, GraphNode, TextDatasetData } from './types'

export const LIBRARY_CATEGORIES = ['All datasets', 'Regression', 'Classification', 'Text and sequences', 'Images', 'Attention and probabilities', 'Custom data'] as const
export type LibraryCategory = typeof LIBRARY_CATEGORIES[number]
export type FootballCoverage = 'football' | 'mixed' | 'benchmark'
export type FootballInput = 'counts' | 'tokens' | 'separate' | 'answer-token'
export interface LibraryPreset {
  id: string
  label: string
  description: string
  asset?: keyof typeof assets
  builtin?: DatasetKind
  task?: CustomCsvData['task']
  football?: { task: 'narrative' | 'questions'; coverage?: FootballCoverage; input?: FootballInput; targets?: 'last' | 'all' }
}
export interface LibraryFamily {
  id: string
  label: string
  category: LibraryCategory
  description: string
  source: string
  sourceUrl?: string
  preparation: string
  presets: LibraryPreset[]
}
export interface LibrarySelection { dataset: DatasetKind; customCsv?: CustomCsvData; textData?: TextDatasetData; datasetLibraryPreset?: string }

const footballPresets: LibraryPreset[] = [
  { id: 'narrative-last', asset: 'narrative-last', label: 'Narrative generation', description: 'Stories end with a statement about a person, the football, or possession. Predict one next token after each context window. No questions.', football: { task: 'narrative', targets: 'last' } },
  { id: 'narrative-all', asset: 'narrative-all', label: 'Narrative generation · every position', description: 'The identical stories and split, with a next-token target at every input position. No questions.', football: { task: 'narrative', targets: 'all' } },
  { id: 'football-counts', asset: 'football-counts', label: 'Football location · word counts', description: 'A smaller question set: count the words in each story and question, then predict the football’s room.', football: { task: 'questions', coverage: 'football', input: 'counts' } },
  { id: 'football-tokens', asset: 'football-tokens', label: 'Football location · ordered tokens', description: 'The same smaller question set as word counts, retaining the order of all 23 story and question tokens.', football: { task: 'questions', coverage: 'football', input: 'tokens' } },
  { id: 'football-mixed', asset: 'football-mixed', label: 'Mixed-question classifier', description: 'Train and evaluate on questions about both people’s locations, the football’s location, and who holds it. Story and question have separate token streams and padding masks.', football: { task: 'questions', coverage: 'mixed', input: 'separate' } },
  { id: 'football-benchmark', asset: 'football-benchmark', label: 'Mixed training · football benchmark', description: 'Train on all four question types; evaluate only where the football is. Separate story and question inputs. These held-out football questions match the smaller question set.', football: { task: 'questions', coverage: 'benchmark', input: 'separate' } },
  { id: 'football-qa', asset: 'football-qa', label: 'Answer-token fine-tuning', description: 'The same questions and split as the football benchmark. Feed the story and question together and predict one answer token. The answer is excluded from the input.', football: { task: 'questions', coverage: 'benchmark', input: 'answer-token' } },
]
function builtins(kinds: DatasetKind[]): LibraryPreset[] {
  return kinds.map(kind => {
    const data = DATASET_OPTIONS.find(item => item.kind === kind)!
    return { id: kind, builtin: kind, label: data.label, description: data.description ?? 'A small synthetic dataset for inspecting model calculations and training.' }
  })
}
export const DATASET_LIBRARY: LibraryFamily[] = [
  { id: 'auto-mpg', label: 'Auto MPG', category: 'Regression', description: 'Predict fuel economy from vehicle weight and model year.', source: 'UCI Auto MPG · CC BY 4.0', sourceUrl: 'https://doi.org/10.24432/C5859H', preparation: 'Original numeric units. Fixed row splits. The smaller variant retains the held-out rows and reduces the training sample for overfitting experiments.', presets: [
    { id: 'auto-mpg', asset: 'auto-mpg', task: 'regression', label: 'Full sample', description: '398 vehicles with weight, model year, and fuel economy.' },
    { id: 'auto-mpg-small', asset: 'auto-mpg-small', task: 'regression', label: 'Small training sample', description: '24 training vehicles and the same 80 held-out vehicles, for exploring regularization.' },
  ] },
  { id: 'concrete', label: 'Concrete strength', category: 'Regression', description: 'Predict compressive strength from eight mixture and age measurements.', source: 'I-Cheng Yeh · UCI Concrete Compressive Strength · CC BY 4.0', sourceUrl: 'https://doi.org/10.24432/C5PK67', preparation: '1,030 rows in original units. Identical feature vectors stay in the same split. Standardization is a model operation.', presets: [{ id: 'concrete', asset: 'concrete', task: 'regression', label: 'Concrete strength', description: 'Eight numeric features and one strength target.' }] },
  { id: 'breast-cancer', label: 'Breast cancer measurements', category: 'Classification', description: 'Classify benign and malignant samples using mean radius and texture.', source: 'UCI Wisconsin Diagnostic Breast Cancer · CC BY 4.0', sourceUrl: 'https://doi.org/10.24432/C5DW2B', preparation: '569 samples with two original numeric features. Fixed stratified split; malignant is 1 and benign is 0.', presets: [{ id: 'breast-cancer', asset: 'breast-cancer', task: 'binary-classification', label: 'Radius and texture', description: 'Two numeric features and a binary target.' }] },
  { id: 'iris', label: 'Iris flowers', category: 'Classification', description: 'Classify three flower species using four sepal and petal measurements.', source: 'R. A. Fisher · UCI Iris · CC BY 4.0', sourceUrl: 'https://doi.org/10.24432/C56C76', preparation: 'Measurements in centimeters. Fixed split with 40 training and 10 held-out examples per species. Labels: setosa, versicolor, virginica.', presets: [{ id: 'iris', asset: 'iris', task: 'classification', label: 'Three species', description: '150 flowers, four measurements, three classes.' }] },
  { id: 'rings', label: 'Concentric rings', category: 'Classification', description: 'Explore nonlinear decision boundaries with three classes in two dimensions.', source: 'Synthetic data generated for Neural Canvas.', preparation: '900 noisy points: lower outer ring, upper outer ring, and inner ring. Fixed training and held-out rows.', presets: [{ id: 'rings', asset: 'rings', task: 'classification', label: 'Three ring classes', description: 'Two numeric coordinates and a three-class target.' }] },
  { id: 'imdb', label: 'Movie reviews', category: 'Text and sequences', description: 'Predict positive or negative sentiment from word-count features.', source: 'Stanford Large Movie Review Dataset · Maas et al., ACL 2011', sourceUrl: 'https://ai.stanford.edu/~amaas/data/sentiment/', preparation: 'Balanced subset: 1,200 training and 400 held-out reviews. Training-fitted vocabulary of 1,000 entries; use up to 128 text tokens per review.', presets: [{ id: 'imdb', asset: 'imdb', label: 'Word-count sentiment', description: 'A prepared sentiment dataset with vocabulary and document splits already fixed.' }] },
  { id: 'football', label: 'Football stories', category: 'Text and sequences', description: 'Follow two people and a football through six rooms. Generate stories or answer questions about locations and possession.', source: 'Synthetic stories generated for Neural Canvas.', preparation: 'Five-event histories. Related room layouts and actor/event-order variants stay together in one split. Mixed questions ask where the football, Liam, and Nora are, and who has the football. Held-out data used to choose settings is validation data.', presets: footballPresets },
  { id: 'wine', label: 'Wine cultivars', category: 'Classification', description: 'Classify three wine cultivars using thirteen chemical measurements.', source: 'UCI Wine · CC BY 4.0 · distributed by scikit-learn', sourceUrl: 'https://doi.org/10.24432/C5PC7J', preparation: '178 samples in original numeric units; cultivar IDs 0–2. Fixed stratified split: 142 training and 36 held-out samples.', presets: [{ id: 'wine', asset: 'wine', task: 'classification', label: 'Three cultivars', description: 'Thirteen numeric features and one three-class target.' }] },
  { id: 'diabetes', label: 'Diabetes progression', category: 'Regression', description: 'Predict disease progression one year later from ten baseline measurements.', source: 'Efron et al. diabetes data · distributed by scikit-learn', sourceUrl: 'https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset', preparation: '442 samples with raw baseline measurements, including age, sex coding, BMI, blood pressure, and six serum measures. No standardization is pre-applied. Fixed split: 354 training and 88 held-out samples.', presets: [{ id: 'diabetes', asset: 'diabetes', task: 'regression', label: 'Raw baseline measurements', description: 'Ten numeric features and a continuous progression target.' }] },
  { id: 'penguins', label: 'Palmer penguins', category: 'Classification', description: 'Classify Adelie, Chinstrap, and Gentoo penguins from four body measurements.', source: 'palmerpenguins · Horst, Hill and Gorman · CC0', sourceUrl: 'https://allisonhorst.github.io/palmerpenguins/', preparation: '342 complete measurement records; two records with missing numeric measurements are omitted. Only bill length/depth, flipper length, and body mass are inputs. Fixed stratified split: 273 training and 69 held-out records.', presets: [{ id: 'penguins', asset: 'penguins', task: 'classification', label: 'Body measurements', description: 'Four numeric features and three species classes.' }] },
  { id: 'california-housing', label: 'California housing', category: 'Regression', description: 'Predict median house value from census block-group measurements.', source: '1990 California census housing · StatLib data, mirrored in Hands-On Machine Learning', sourceUrl: 'https://github.com/ageron/handson-ml2/tree/master/datasets/housing', preparation: '20,433 complete records; 207 rows missing total bedrooms are omitted. Eight numeric inputs in original units, including longitude/latitude, block totals, and median income; ocean proximity is excluded. Target values are in dollars. Fixed split: 16,346 training and 4,087 held-out records.', presets: [{ id: 'california-housing', asset: 'california-housing', task: 'regression', label: 'Complete numeric records', description: 'A larger tabular regression dataset with eight features. House values retain the source ceiling.' }] },
  { id: 'regression-examples', label: 'Simple regression examples', category: 'Regression', description: 'Small line, cubic, and plane datasets for building a first model.', source: 'Synthetic examples included with Neural Canvas.', preparation: 'Included deterministic data and default splits.', presets: builtins(['line-1d', 'cubic-1d', 'plane-2d']) },
  { id: 'classification-examples', label: 'Decision boundaries', category: 'Classification', description: 'Thresholds, circle versus center, a parabola, XOR, and neuron basics.', source: 'Synthetic examples included with Neural Canvas.', preparation: 'Included deterministic data and default splits.', presets: builtins(['threshold-1d', 'circle-center', 'parabola-boundary', 'xor', 'neuron-basics']) },
  { id: 'digits', label: 'Handwritten digits', category: 'Images', description: 'Classify small 8 × 8 images into ten digit classes.', source: 'UCI Optical Recognition of Handwritten Digits · CC BY 4.0', sourceUrl: 'https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits', preparation: '500 included images: 400 training and 100 held-out. Pixels normalized to 0–1. This is not MNIST.', presets: builtins(['digits-8x8']) },
  { id: 'token-patterns', label: 'Repeating token patterns', category: 'Text and sequences', description: 'Learn next-token prediction with color cycles or counting sequences.', source: 'Synthetic examples included with Neural Canvas.', preparation: 'Token IDs, position IDs, and aligned next-token targets. Sequence lengths 5 and 8 stay held out.', presets: builtins(['color-cycle', 'counting']) },
  { id: 'attention-examples', label: 'Attention message lookups', category: 'Attention and probabilities', description: 'Inspect how queries select and combine message values.', source: 'Synthetic examples included with Neural Canvas.', preparation: 'Included query, key, and value tensors with desired-message targets.', presets: builtins(['attention-query', 'attention-sequence']) },
  { id: 'class-scores', label: 'Class scores and softmax', category: 'Attention and probabilities', description: 'Turn score vectors into class probabilities and predictions.', source: 'Synthetic examples included with Neural Canvas.', preparation: 'Four-class score vectors with fixed training and held-out examples.', presets: builtins(['class-scores']) },
]
export function libraryPreset(id: string): LibraryPreset {
  const preset = DATASET_LIBRARY.flatMap(family => family.presets).find(item => item.id === id)
  if (!preset) throw Error('Unknown dataset preset.')
  return preset
}
export async function loadLibraryDataset(id: string, signal?: AbortSignal): Promise<LibrarySelection> {
  const preset = libraryPreset(id)
  if (preset.builtin) return { dataset: preset.builtin, datasetLibraryPreset: id }
  if (!preset.asset) throw Error('This dataset has no source file.')
  const asset = assets[preset.asset]
  const response = await fetch(`${import.meta.env.BASE_URL}datasets/${asset.file}`, { signal })
  if (!response.ok) throw Error('Could not load this dataset. Check your connection and try again.')
  if (asset.file.endsWith('.csv')) {
    const customCsv = { ...parseCustomCsv(await response.text(), asset.file), task: preset.task! }
    analyzeCustomCsv(customCsv)
    return { dataset: 'custom-csv', customCsv, datasetLibraryPreset: id }
  }
  const textData: unknown = await response.json()
  if (!isTextDatasetData(textData)) throw Error('This prepared dataset is invalid.')
  return { dataset: 'custom-text', textData, datasetLibraryPreset: id }
}
export function libraryPreview(selection: LibrarySelection) {
  const node: GraphNode = { id: 'dataset-preview', type: 'dataset', label: 'Dataset', position: { x: 0, y: 0 }, params: { ...selection } }
  const data = datasetForNode(node), examples = datasetExamplesForNode(node), first = examples[0]
  const text = selection.textData
  const document = text?.documents[first.documentIndex ?? 0]
  const input = text?.task === 'language' ? first.text : document?.text
  const target = text?.task === 'language' ? decodeTokens(first.target.data, text)
    : document?.label ?? data.classLabels?.[first.target.data[0]] ?? first.target.data.join(', ')
  return {
    train: examples.filter(example => example.split === 'train').length,
    test: examples.filter(example => example.split === 'test').length,
    vocabulary: text?.vocabulary.length ?? data.vocabulary?.length,
    classes: data.classLabels,
    outputs: [...data.featureLabels.map((label, slot) => ({ label, shape: first.features[slot].shape })), { label: data.targetLabel, shape: first.target.shape }],
    input: input?.slice(0, 700), question: document?.question, target,
    numeric: !text ? first.features.map((value, slot) => ({ label: data.featureLabels[slot], values: value.data.slice(0, 12).join(', ') })) : undefined,
  }
}
