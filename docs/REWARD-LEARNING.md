# Learning from verifiable rewards and human preferences

Use **Train → Reward learning** with a categorical model that produces one
answer-logit vector per example. The football answer-token transformer is one
such model. The supported exercise is a one-step contextual bandit with a small
finite answer space. It uses exact expectations over every candidate answer;
it does not sample rollouts or implement PPO, GRPO, or delayed credit assignment.

## Reference and action space

Choose **Reward source**, then **Capture current model as reward reference**.
This saves the current parameter tensors as an immutable reference. Candidates
are the distinct answer IDs in the training split (2–64 candidates). Validation
labels never add candidates. Output probabilities are normalized over these
candidates, rather than the full token vocabulary. Changing the dataset or graph
operations requires a fresh reference. Capturing again discards earlier
preferences, reward weights, and reports; save the earlier project first.

**Reference KL strength** controls how strongly optimization stays near the
reference policy. For context x, candidate a, frozen reward r, current policy p,
and reference policy q, the minimized objective is:

`mean_x sum_a p(a|x) [beta * (log p(a|x) - log q(a|x)) - r(x,a)]`

Only current policy parameters receive gradients. Frozen Params remain fixed.
Reward learning evaluates dropout deterministically. It changes the weights of
the existing model and preserves its graph. **Run epochs** continues to train the
ordinary canvas Loss; reward optimization has its own explicit button.

## Verifiable rewards

Choose **Verifiable answer correctness**. The checker assigns reward 1 to the
dataset's correct answer and 0 to other candidates. All candidate rewards are
available in this bounded exercise. This objective differs from supervised
cross-entropy: it maximizes expected correctness, with a KL penalty, rather than
maximizing the log probability of a supplied target.

## Human preferences

Capture the adapted model again with **Human preference comparisons** selected.
Read each **Training context**, select distinct **Preferred answer** and
**Rejected answer** values, and click **Record human preference**. The context
shows the input tokens, excluding the answer target. Repeating a comparison
replaces the previous judgment on that pair. Recording a new judgment clears an
already fitted reward model, since it no longer describes all recorded data.

**Fit preference reward model** performs 200 deterministic batch-gradient steps
at learning rate 0.5. Its features are the frozen reference policy's candidate
probabilities. For K candidates it learns a K×K weight matrix and K biases:

`score(x,a) = bias[a] + sum_i reference_probability(i|x) * weights[i,a]`

Pairwise logistic loss encourages the preferred answer's score to exceed the
rejected answer's score. A 0.001 L2 gradient applies to the weights. Policy
optimization receives `sigmoid(score)` as a bounded reward. The reference
features and reward weights stay fixed while the policy changes. This compact
reward model can generalize imperfectly and can reward unwanted behavior; the
interface reports exact correctness alongside predicted reward so students can
investigate that discrepancy. Its training loss is a fit diagnostic, not an
independent measure of human preference alignment.

The preferences entered by students are human judgments. Automated fixtures
derived from the answer checker are simulated comparisons and must be described
as such; they do not validate a human-feedback study.

## Optimize, inspect, and save

**Optimize policy with rewards** uses the existing **Epochs per run**,
**Report loss every**, **Examples per update**, learning rate, optimizer, and
gradient norm limit. It uses the Tensor engine and keeps final parameters;
early stopping is not used for this objective. **Evaluate reward policy** reads
the same reports without updating parameters. Stopping a run keeps the state
from before that run.

The reward report contains training and validation expected reward, expected
checker correctness, candidate accuracy, reference KL, and minimized reward
objective. Ordinary **Test** still measures cross-entropy and accuracy against
the full model output vocabulary; its numbers answer a different question.
Validation used to choose settings is not an untouched final test.

**File → Save** includes the reference tensors, candidate IDs, preference pairs,
fitted reward weights, configuration, and last report in `graph.rewardLearning`.
Import validates their bounded numerical structure. Recompute any graded metric
from the graph and canonical data; saved reports and provenance fields are not
evidence that training or human annotation occurred.

After the final reward assignment, save the final transformer. Use **Clear reward
setup** to remove the old reference, judgments, reward weights, and reports while
preserving the current transformer parameters. Retain a frozen copy before
training an SAE on its activations. The steering SAE must match that
exact transformer checkpoint and activation location.

Python/notebook export exports the current graph's weights and ordinary Loss
training loop. The separate reward procedure and saved reference/preference
state are preserved by project JSON, not by that exported training loop.
