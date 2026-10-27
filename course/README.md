# Teach with Neural Canvas

This directory contains the complete current teaching releases for the
[19-assignment course](../README.md#course-assignments): student handouts,
ready-to-upload autograders, grader source, instructor notes, rubrics,
curriculum plans, and validation summaries. No private repository is needed to
teach the course or rebuild its autograders.

Created by David Bachman with Codex.

## Use an assignment

1. Open its folder under [assignments/](assignments/) and read `Instructor/INSTRUCTOR.md`.
2. Read `description.md` on GitHub, or paste `description.html` into Canvas's HTML
   editor; set your own due date and Gradescope link. Students select datasets in Neural Canvas's dataset
   browser. Datasets live in the app's [public/datasets/](../public/datasets/);
   no dataset attachments or assignment-local copies are needed.
3. Upload the ZIP in `Gradescope/` to a 100-point Gradescope Programming
   Assignment, using the base image, memory, and timeout in the instructor guide.
4. Follow the handout's model filenames and the guide's submission/rubric notes.
   Test your own saved models in Gradescope before issuing the assignment.

Students build their own editable models. Later assignments reuse earlier
trained models; preserve those handoffs. A repository push does not update
Canvas or Gradescope. Local runner checks do not certify hosted Gradescope.

## Directory layout

| Location | Contents |
| --- | --- |
| `assignments/*/description.html` | Canonical student handout for Canvas's HTML editor |
| `assignments/*/description.md` | Generated Markdown handout for viewing on GitHub |
| `assignments/*/Gradescope/` | One current Gradescope upload ZIP |
| `assignments/*/source/` | Self-contained bounded grader, runner, and setup |
| `assignments/*/reference/` | Canonical grading data, untrained graph templates, provenance, and package hashes |
| `assignments/*/Instructor/INSTRUCTOR.md` | Publishing, construction, rubric, teaching, and validation notes |
| [docs/REMAINING-SEMESTER-SCHEDULE.md](docs/REMAINING-SEMESTER-SCHEDULE.md) | Current course sequence |
| [docs/ASSIGNMENT-DESIGN.md](docs/ASSIGNMENT-DESIGN.md) | Course design and assessment principles |
| [docs/README.md](docs/README.md) | Current and historical validation reports |
| `scripts/assignments/` | Packaging and handout helpers |

Wiring, untrained templates, autograder logic, thresholds, and pilot summaries
are intentionally public. Complete trained solutions, answer keys, model-producing
pilot tools, fixture-dependent regression tests, historical releases containing
solutions, and future exams remain in the private NeuralCanvas-Coursework repo.
The instructor guides retain original pilot app revisions and reproduction
commands; commands involving pilots, private references, or `test_grader.py` run
there. Those files are optional development evidence, not prerequisites for
teaching. Build your own models when testing a new Gradescope setup.

## Verify and rebuild

From this directory, use Python 3.12:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/validate-course.py
```

The check verifies every upload ZIP against its canonical sources and hashes,
checks the referenced app datasets against their recorded hashes, checks that
templates contain no trained weights, and runs each packaged grader with malformed submissions.
It needs no private fixtures or network downloads. To grade your own models,
extract a ZIP into `source/` in a temporary directory, put saved JSON files in
its sibling `submission/`, then run:

```sh
AUTOGRADER_ROOT=/absolute/path/to/temporary-directory .venv/bin/python /absolute/path/to/temporary-directory/source/runner.py
```

To rebuild all current uploads from committed data without training:

```sh
.venv/bin/python scripts/assignments/package-mpg.py
.venv/bin/python scripts/assignments/package-regularization.py
.venv/bin/python scripts/assignments/package-classification.py
.venv/bin/python scripts/assignments/package-matrices.py
.venv/bin/python scripts/assignments/package-mlp.py
.venv/bin/python scripts/assignments/package-rings.py
.venv/bin/python scripts/assignments/package-word-counts.py
.venv/bin/python scripts/assignments/package-embeddings.py
.venv/bin/python scripts/assignments/package-advanced.py
```

Packaging refreshes the ZIP and `reference/package-sha256.json`. Preserve issued
datasets and rubrics; do not redownload or refit data to run checks.

The Markdown descriptions and deployed student pages are generated from the
canonical HTML handouts. From the app root, run
`python course/scripts/sync-handouts.py` after editing handouts,
or add `--check` to verify they are current. The same command refreshes the
description hashes in each release manifest without rebuilding grader ZIPs.
Instructor guides and graders are available through GitHub and are not copied
into the app's static site.
