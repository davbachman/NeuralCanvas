# Populated-tensor canvas navigation

## Reproduction and measurement

The repository fixture `src/test/canvasPerformanceModel.ts` uses 16 token IDs,
a vocabulary of 27, a `[27,8]` embedding table, flattening to `[1,128]`, and
an MLP with `[128,32]` and `[32,6]` weights, biases, ReLU, cross-entropy and a
separate Softmax branch. It runs forward and backward to populate values,
gradients and derivative caches. The alternative fixture replaces One-hot +
Matmul with Embedding and uses identical weights. A regression test compares
predictions, loss and parameter gradients.

To reproduce from the repository root:

```sh
node scripts/browser/canvas-performance-fixtures.mjs
npm run build
npm run preview -- --host 0.0.0.0
# In another terminal, with Playwright installed:
PLAYWRIGHT_MODULE=/path/to/node_modules/playwright \
  node scripts/browser/canvas-performance.cjs measured
```

`CHROMIUM` defaults to `/usr/bin/chromium`; `CANVAS_URL` defaults to
`http://localhost:4173/NeuralCanvas/`. The runner writes JSON measurements and
Chrome traces into ignored `output/`. It checks that the imported fixture
actually loaded before measuring. No external datasets are needed.

Measurements use headless Chromium, a 1600×1000 viewport, the production app,
the Details sidebar, fresh browser contexts, and DevTools CPU throttling at
1× and 4×. Each case performs six secondary-button drag gestures (25 moves
per gesture) and then 150 alternating wheel-zoom events. Both variants retain
all populated outputs and gradients. Runs are sequential, without concurrent
builds/tests. The number of input events is fixed; elapsed time can increase
when the browser cannot keep up. Frame intervals come from requestAnimationFrame;
long tasks, script/layout/style durations and paint/GC traces come from browser
performance APIs and CDP. Large JSON parse/stringify and IndexedDB put calls are
measured separately; their times overlap script/task time and must not be added
to it. GC and paint trace durations likewise should not be added to task time.

## Diagnosis and changes

- Secondary-button panning called `setViewport` for each pointer move, and each
  call emitted `onMoveEnd`. Persisting all those intermediate cameras rebuilt
  trace graphs, parameter editors, node/edge descriptions and their DOM. A drag
  now commits once on release (or lost capture). Recovery still receives the live
  viewport through a ref, including for pagehide during a gesture.
- Full tensor descriptions and wire intensities now use weak caches keyed by
  immutable tensor values. Compact previews read only the displayed entry;
  shape badges do not validate or copy the buffer. New values invalidate the
  caches naturally, and weak keys do not retain discarded training graphs.
- Model derivations exclude the camera. Stable callbacks keep node/edge data
  reusable without capturing stale edit handlers. Grouped scenes still update
  zoom-dependent reveals, port geometry, accessible hit targets and breadcrumbs.
- An inherited selection-outline custom property on the canvas root caused
  broad style recalculation on every wheel tick. Its update now applies only to
  the selection outline, including outlines mounted later by React Flow.
- Recovery keeps a validated full model snapshot plus a small viewport patch
  tied to that snapshot's timestamp. Navigation patches are coalesced while
  moving, with a bounded periodic flush. Model edits and training checkpoints
  still take the full validation/save path; lifecycle flushes bypass navigation
  deferral. During training, camera-only writes use the newest durable
  checkpoint at execution time, so they cannot roll back newer weights. A full
  write atomically clears the old camera patch. No tensors,
  gradients, outputs or caches are removed to obtain these improvements.

## Results — 2026-10-17

Baseline: `e16b40e`; optimized: this change. Chromium 151.0.7922.173 on Linux, same
machine and configuration for both builds. These are single measured runs per
configuration, not a cross-device frame-rate guarantee. The One-hot graph has
17 nodes / 16 edges; Embedding has 16 nodes / 15 edges (a separate Target node
is included in both).

| Model | CPU slowdown | Navigation | p95 frame interval, before → after (ms) | Frames >50 ms, before → after | JavaScript, before → after (ms) |
|---|---:|---|---:|---:|---:|
| One-hot | 1× | pan | 66.7 → 16.8 | 41 → 0 | 4175 → 274 |
| One-hot | 1× | zoom | 16.8 → 16.8 | 0 → 0 | 891 → 203 |
| One-hot | 4× | pan | 316.7 → 16.8 | 155 → 2 | 20623 → 1332 |
| One-hot | 4× | zoom | 100.0 → 16.8 | 147 → 1 | 4579 → 871 |
| Embedding | 1× | pan | 66.6 → 16.7 | 25 → 0 | 3981 → 246 |
| Embedding | 1× | zoom | 16.8 → 16.7 | 1 → 0 | 981 → 173 |
| Embedding | 4× | pan | 300.1 → 16.8 | 157 → 1 | 19765 → 1216 |
| Embedding | 4× | zoom | 100.0 → 16.7 | 133 → 1 | 4980 → 927 |

The breakdown below uses the **4× throttled** runs. Times are totals for the
same sequence of input events, in milliseconds. Style recalculation is listed
separately from layout and paint. GC is the sum of top-level MinorGC/MajorGC
trace spans. Large JSON timings include all calls over 50 KB, including editor
formatting as well as recovery; they are not a recovery-only measurement.

| Model / navigation | Layout before → after | Style before → after | Paint before → after | GC before → after | Large JSON before → after | Recovery puts before → after |
|---|---:|---:|---:|---:|---:|---:|
| onehot / pan | 415 → 166 | 8322 → 82 | 7299 → 1158 | 354 → 25 | 568 → 0 | 13 → 0 |
| onehot / zoom | 190 → 165 | 5670 → 141 | 3489 → 881 | 85 → 25 | 49 → 0 | 1 → 1 |
| embedding / pan | 424 → 139 | 8002 → 86 | 6971 → 1002 | 365 → 33 | 532 → 0 | 12 → 0 |
| embedding / zoom | 197 → 172 | 5678 → 153 | 3327 → 951 | 98 → 16 | 141 → 0 | 5 → 1 |

CPU stack sampling on the baseline throttled One-hot pan attributed about
3.43 seconds inclusively to `formatFullTensor` and 0.49 seconds to
`writeRecovery` (clone, validation and serialization). The rest included React
render/commit work and browser work. These sampled durations overlap the totals
above. Recovery was a contributor, not the principal cause. Afterward, all
measured navigation runs made zero large JSON calls; any recovery writes were
small viewport patches. The final camera is also flushed after navigation or
on pagehide, so zero puts during a short pan does not mean recovery is disabled.

Removing root-level CSS inheritance alone reduced zoom style work but left
React work on every flat-canvas camera tick. Letting React Flow own that camera,
with a direct update for the selection outline, removed that remaining work.
Painting still has a measurable cost under throttling, and isolated long frames
remain; the repeated navigation stalls are substantially reduced.

## Verification

- Full suite: 914 passed, 29 skipped. Final focused canvas/recovery checks:
  136 passed. Lint and production build pass (existing bundle-size advisory).
- Tests cover bounded compact reads, description/intensity reuse, model/viewport
  separation, stable node and edge data, secondary-drag commit coalescing,
  nested pan/zoom and accessibility, current callback behavior, selection-outline
  scaling, live-camera lifecycle flushing, bounded recovery deferral, training
  checkpoints, full-save invalidation and viewport restoration.
- A production-browser check selected Softmax, panned and wheel-zoomed, inspected
  Data, saved recovery and reloaded. Viewport, selection, parameters, node/edge
  outputs, gradients and derivative caches restored exactly, with no page errors.
- The fixture's numerical regression compares One-hot + Matmul with Embedding:
  predictions, loss and parameter gradients are identical. Numerical execution
  is unchanged.

Raw traces and JSON results are in `output/baseline-final-*` and
`output/verified-*` in the measurement workspace. The scripts above recreate the
fixtures and measurements without external coursework files.
