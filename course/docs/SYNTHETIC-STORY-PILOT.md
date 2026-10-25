# Synthetic narrative generation: feasibility pilot

For the current synthetic-narrative Assignments 11–20, see [NARRATIVE-REVISION.md](NARRATIVE-REVISION.md). The measurements and earlier sequence below are historical evidence; they do not validate the replacement dataset.

2026-10-19. Private evidence, not an assignment release or grading rubric.

## Decision and scope

The instructor authorized a complete redesign of Assignments 11–20, with changes
to unpublished Assignments 7–10 if useful. Assignments 11–19 should focus on
**generative models**; they need not relate to question answering. A question-answer
fine-tuning exercise is an optional final extension, contingent on evidence.
Training targets are about five minutes per model, with up to ten acceptable late
in the sequence. Issued Assignments 1–6 and all current handouts/graders remain
unchanged by this pilot.

The instructor clarified that an empirical improvement in every lesson is not
required. Aim for a clear advantage to discuss: accuracy, speed, parameter or
memory efficiency, parallelizability, stability, or a new capability. Distinguish
measured benefits from architectural arguments and state their tradeoffs. A lack
of accuracy gains alone is not a reason to eliminate a lesson.

**Finding:** synthetic narratives give us a measurable context/coherence challenge
and viable browser runtimes. They do not yet justify transplanting all ten Alice
assignments or claiming a benefit for every architectural addition. In particular,
these tests do not establish independent accuracy benefits for normalization or
separate attention-only/residual lessons. Other benefits would need appropriate
evidence or an explicit architectural explanation. Instruction tuning is
technically feasible but has not demonstrated a reliable final-accuracy advantage
from narrative pretraining.

## Narrative tasks

The smaller corpus has 256 training stories and 128 held-out stories. Each gives
two people randomly chosen destinations, then continues with two
statements about their locations, in varying person order. There are 20 vocabulary
tokens including unknown/padding tokens, a 26-token input, and 27 tokens per story.
For example:

> Maya went to the garden. Nora went to the hall. Now Nora is in the hall.
> Now Maya is in the garden. End.

The harder corpus has four movement events, including repeated moves for each
person, before its two concluding statements: 1,024 training stories, 256 held-out
stories, and 38-token inputs. A separate variant restricts the cast to two names.
All three corpora are **narrative language-modeling data, without questions**.
These are deliberately small feasibility tasks. Richer stories, variable story
lengths, and additional story templates have not yet been validated.

Complete ordered event histories are disjoint across training and validation;
changing the order of concluding statements cannot move the same history across
splits. Generation seeds, source histories, target positions, and independently
tracked location truth are retained. Vocabularies use training narratives only.
The held-out split is validation during this exploration, not an untouched final
benchmark. No arbitrary future movement is treated as uniquely predictable.

## Browser evidence

These are actual Neural Canvas Tensor CPU runs in Chromium 151.0.7922.173 on clean
app commit `2b6b96d8d07ee52e0beca3c3432f9f4e5819af79`. They invoke the browser training
engine without the full training UI callbacks. Timings include training reports;
some runs overlapped other pilots. They are cloud measurements, not student-laptop
guarantees. No native hardware GPU benchmark is claimed.

The primary metric below is next-word correctness at the **two concluding location
words**, with preceding words supplied. It avoids giving credit primarily for
predicting repeated syntax. Each attention run used Adam, learning rate 0.01,
batch 32 stories, clipping 1, no early stopping, 40 epochs, and reports every 10
on the smaller corpus. Width is 16; the two-head models use two width-8 heads in
each of two blocks. One-head and two-head runs share initial parameter values
for each seed and have equal parameter counts. Two heads cost more computation;
equal parameters are not equal runtime.

| Model | Seeds | Concluding-location accuracy | Training time |
|---|---|---|---|
| Eight-word-context MLP | 137 | 31.6% | 40.6 s |
| Two blocks, one head each | 137 / 211 / 307 | 54.7% / 79.7% / 73.4% | 77.3 / 96.0 / 79.2 s |
| Two blocks, two heads each | 137 / 211 / 307 | 79.3% / 75.4% / 80.5% | 100.6 / 101.1 / 117.9 s |

The MLP uses ordinary eight-word sliding contexts, 64 hidden units, and 10 epochs
on 4,864 training windows; its first eight story words are not training targets.
All conclusion words are included. Attention uses the complete 26-word sequence
and supervision at every position. This comparison combines context length and
training organization; **it does not isolate attention against a full-context MLP**.
A full-context MLP in the PyTorch screen reached 79.3–83.2% at epoch 40, with 28,308
parameters, versus 5,428 for either two-block attention model. That screening uses
vectorized MLP windows and is not an app runtime benchmark. Attention's promising
advantages here include parameter efficiency and coherent longer continuations,
not an established inability of MLPs to solve the task.

Across the three matched browser seeds, two heads improved mean location accuracy
from 69.3% to 78.4%, but one seed became worse. This is encouraging small-sample
evidence for a controlled investigation, not a promise that adding heads always
helps. Broader PyTorch screening also exposed weak initialization-dependent results
for attention-only and single-block normalized models.

On the harder repeated-movement corpus, the native two-block/two-head run took
246.9 seconds for ten epochs and reached 83.4% overall next-word accuracy but only
55.9% concluding-location accuracy. Larger/longer PyTorch screens did not reliably
solve its held-out state tracking. Starting with this harder task would conceal
an unresolved learning problem behind fluent sentence templates.

## Free-running narrative continuations

A separate check supplies only the two opening movement sentences and lets each
model generate the rest. No correct summary words are fed back. We use 64 fixed
held-out prefixes, up to 15 new words, and stop at `end`. A fully consistent
continuation must complete the two expected narrative clauses, mention both
original people exactly once, and place each in the correct room.

At greedy decoding:

| Model | Fully consistent continuations |
|---|---|
| Eight-word-context MLP, seed 137 | 1.6% |
| Two blocks, one head each, three seeds | 23.4% / 46.9% / 28.1% |
| Two blocks, two heads each, three seeds | 46.9% / 46.9% / 54.7% |

The attention models completed grammatical continuations in all these greedy
cases, while many continuations were factually inconsistent. This distinction
between grammatical fluency and faithful continuation is useful and observable.
The MLP completed the expected grammar in 37.5% of these cases.

Temperatures 0.7, 1.0, and 1.3 were also sampled with recorded seeds. Complete
results and concrete generated continuations are in
`pilots/story-language/generation.json`. These are controlled continuations,
not an evaluation of unrestricted whole-story generation from a single word.
Independent generation was compared with the app's actual browser generation
function on 18 prefixes across the MLP, one-head and two-head models.

## Optional final adaptation

A separate experiment changed the task to predicting a single answer word after
a narrative and a `where is [person] ?` prompt. A final-row Slice and the existing
last-target dataset mode implement answer-only loss; no new app operation is
needed. Two new tokens were added for this experimental transfer. A released
exercise should supply a compatible vocabulary in advance so it does not become
a parameter-resizing exercise.

The native CPU capability check used 128 training histories (256 questions),
held-out narrative histories, five epochs, and rate 0.003. The pretrained model
reached 57.4% after one epoch and 55.9% after five; the fresh model reached 17.6%
and 51.6%, respectively. Training took approximately 12 and 10 seconds. Thus
pretraining helped initial adaptation in this particular run, but the endpoint
remains weak.

Longer PyTorch checks with three adaptation seeds, more data, and a smaller
fine-tuning learning rate did **not** establish a dependable final advantage.
Some fresh models overtook the pretrained model. Those three adaptation seeds
share a single pretrained checkpoint, so they are not three independent
pretraining replications. No comparison with Assignment 10 has yet been validated.
This extension should remain optional and should not determine the narrative
curriculum.

## Recommendation before rewriting

1. Keep Assignments 11–19 generative. Begin with short coherent narratives and a
   clear local-context limitation; introduce longer context and attention through
   an independently graded continuation/coherence challenge.
2. Treat a working transformer block as a coordinated construction. Do not promise
   a separate accuracy win from each residual connection or normalization step.
   Those components can support separate lessons about optimization or scaling
   when the benefit is demonstrated or clearly explained. Consider combining
   lessons when neither a useful advantage nor sufficient student work emerges.
3. Retain a controlled head-count comparison as a candidate, but expand validation
   before setting thresholds. There is a useful mean/consistency signal here,
   with an explicit counterexample seed.
4. Use generation and sampling as substantive graded work: saved generative models
   can be evaluated through seeded continuations, grammar, factual consistency,
   and coverage. These metrics require their own robust grader and multi-seed
   thresholds; none have been released in this task.
5. Before choosing a ten-day sequence, pilot richer/variable-position narratives
   and identify a defensible advantage for each proposed lesson. Measure empirical
   claims and distinguish structural benefits, such as parallel training across
   token positions, from browser speed claims; autoregressive generation still
   proceeds token by token. The current small corpus alone does not validate all
   proposed advantages or provide enough independent work for ten assignments.

No development request is currently necessary for the tested workflows. App
changes, if a later exercise requires them, should be requested separately rather
than silently implemented in the app repository.

## Reproduction and validation

Source scripts: `scripts/pilots/story-language/`. Fixtures, complete checkpoints,
settings, epoch curves, failures of proposed comparisons, and independent checks:
`pilots/story-language/`. The private evidence is separate from `assignments/`;
there are no alternate student releases.

- `prepare.py`: seeded corpora and state-tracking truth. Defaults create the harder
  corpus. Use `--moves 1 --train 256 --test 128 --out pilots/story-language/static-location`
  for the smaller task; `--people 2` creates the reduced-cast variant.
- `screen.py`: PyTorch architectural screening, including all baseline families,
  widths/depths/heads, and multiple seeds. Per-directory `config.json` records each
  run. These timings do not establish browser performance.
- `finetune.py`: optional answer-only adaptation and fresh controls. Its vocabulary
  expansion and optimizer choices are explicitly recorded.
- `app-model.mjs`, `browser-pilot.mjs`: editable app graphs and actual browser-engine
  training. Each browser `report.json` includes the complete options object.
- `prepare-short-context.py`: native short-window baseline on the same narratives.
- `prepare-qa.mjs`: optional matched pretrained/fresh app models and QA data.
- `validate.py`: verifies disjoint histories, training-only vocabulary, and all
  truth labels; independently recomputes saved browser predictions/losses; checks
  PyTorch/NumPy logit agreement and causal prefix invariance where applicable.
- `generation.py`, `check-generation.mjs`: free-running continuation scoring and
  comparison with actual browser generation.

Use the repository virtual environment for NumPy. PyTorch is an additional private
screening dependency, not a student or autograder requirement; its version is in
each screen checkpoint. Browser scripts use `NEURALCANVAS_ROOT` pointing to the
recorded app checkout, `PLAYWRIGHT_ROOT` pointing to an installed Playwright package,
Chromium at `/usr/bin/chromium`, and `APP_URL` for that checkout's Vite server
(default `http://127.0.0.1:5186/NeuralCanvas/`). For example:

```sh
NEURALCANVAS_ROOT=/path/to/app PLAYWRIGHT_ROOT=/path/to/playwright-package \
  node scripts/pilots/story-language/browser-pilot.mjs config.json /tmp/story-run
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/pilots/story-language/validate.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/pilots/story-language/generation.py
PLAYWRIGHT_ROOT=/path/to/playwright-package \
  node scripts/pilots/story-language/check-generation.mjs
```

Do not infer training time or grading thresholds from the independent Python
checks. No hosted Gradescope test, release packaging, or external publication was
performed or needed for this private feasibility pilot.
