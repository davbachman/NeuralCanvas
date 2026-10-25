"""Independently grade two logistic classifiers; no student code is executed."""
import json
from pathlib import Path
import sys
import numpy as np
from graph_eval import graph_from_file, compile_graph, require, tensor, ZERO, X, Z

VERSION = 'classification-03-v1'
PART_POINTS = {'one_feature': 50, 'two_features': 50}


def evaluate(poly, x):
    return sum((c * x[:, 0]**p[0] * x[:, 1]**p[1] for p, c in poly.terms.items()), np.zeros(len(x)))


def model_result(path, part, data):
    tests, metrics = [], {}
    weights = {'saved model and dataset': 5, 'model form and scaling': 25,
               'training BCE': 10, 'validation BCE': 5, 'validation accuracy': 5}
    def add(name, ok, message):
        tests.append(dict(name=f'{part} — {name}', score=weights[name] if ok else 0,
                          max_score=weights[name], output=message, visibility='visible'))
    try:
        graph = graph_from_file(path, data)
        add('saved model and dataset', True, 'Original dataset, target encoding, and split verified.')
    except Exception as error:
        for name in weights:
            add(name, False, str(error) if name == 'saved model and dataset' else 'Requires a readable saved model and original dataset.')
        return dict(tests=tests, metrics=metrics)
    try:
        width = 1 if part == 'one_feature' else 2
        poly, params, used, _ = compile_graph(graph)
        require(used == set(range(width)), 'Use mean_radius only for Part A; mean_radius and mean_texture for Part B.')
        powers = [X, Z][:width]
        require(set(poly.terms) <= {ZERO, *powers}, 'The logit must be linear in the assigned inputs.')
        require(len(params) == width + 1, 'Use one separate scalar Param per feature and one intercept.')
        nodes = {n['id']: n for n in graph['nodes']}
        loss = next(n for n in graph['nodes'] if n['type'] == 'loss')['params']
        require(loss.get('regularization', 'none') == 'none' or loss.get('regularizationStrength', 0) == 0,
                'Use Penalty: None.')
        x = np.array([[r['radius'], r['texture']] for r in data['rows']])
        y = np.array([r['malignant'] for r in data['rows']])
        train = np.array([r['split'] == 'train' for r in data['rows']])
        scaled = (x - x[train].mean(0)) / x[train].std(0)
        A = np.column_stack([np.ones(len(x)), scaled[:, :width]])
        coefficients, matched = np.zeros(width + 1), set()
        for key, index in params:
            value = tensor(nodes[key]['params'].get('value', .5))
            require(value.size == 1 and index == 0, 'Use separate scalar Param blocks.')
            original = value.flat[0].terms.get(ZERO, 0)
            plus = compile_graph(graph, (key, index, original + .25))[0]
            minus = compile_graph(graph, (key, index, original - .25))[0]
            require(all(set(p.terms) <= {ZERO, *powers} for p in [plus, minus]), 'Do not add nonlinear features.')
            direction = evaluate(plus - poly, x) / .25
            require(np.allclose(direction, evaluate(poly - minus, x) / .25, atol=2e-5, rtol=2e-5),
                    'Parameters must be independent linear coefficients.')
            matches = [k for k in range(width + 1) if np.allclose(direction, A[:, k], atol=2e-5, rtol=2e-5)]
            require(len(matches) == 1 and matches[0] not in matched,
                    'Each coefficient must multiply its own training-standardized feature (or 1 for the intercept). Fit on 456 training rows.')
            k = matches[0]
            matched.add(k)
            coefficients[k] = original
        logits = evaluate(poly, x)
        require(np.allclose(logits, A @ coefficients, atol=1e-5, rtol=1e-5), 'Do not add fixed fitted coefficients.')
        add('model form and scaling', True, 'Correct standardized linear logit, matching BCE wiring, and no penalty.')
    except Exception as error:
        for name in weights:
            if name != 'saved model and dataset':
                add(name, False, str(error) if name == 'model form and scaling' else 'Requires the assigned model and scaling.')
        return dict(tests=tests, metrics=metrics)
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -700, 700)))
    if loss.get('loss') == 'binary-cross-entropy-with-logits':
        # Stable BCE with internal sigmoid, matching the new app loss.
        losses = np.maximum(logits, 0) - y * logits + np.log1p(np.exp(-np.abs(logits)))
    else:
        # Preserve the app's clipping for saved legacy probability BCE models.
        p = np.clip(probabilities, 1e-7, 1-1e-7)
        losses = -(y * np.log(p) + (1-y) * np.log1p(-p))
    tr, val = float(losses[train].mean()), float(losses[~train].mean())
    accuracy = float(np.mean((probabilities[~train] >= .5) == y[~train]))
    train_limit, val_limit = (.31, .29) if width == 1 else (.28, .26)
    metrics = dict(train=tr, validation=val, accuracy=accuracy, thresholds={})
    for threshold in [.25, .5, .75]:
        predicted, actual = probabilities[~train] >= threshold, y[~train].astype(bool)
        metrics['thresholds'][str(threshold)] = dict(
            truePositive=int(np.sum(predicted & actual)), falsePositive=int(np.sum(predicted & ~actual)),
            trueNegative=int(np.sum(~predicted & ~actual)), falseNegative=int(np.sum(~predicted & actual)))
    threshold_feedback = '\n'.join(
        f"Threshold {t}: TP={c['truePositive']}, FP={c['falsePositive']}, TN={c['trueNegative']}, FN={c['falseNegative']}"
        for t, c in metrics['thresholds'].items())
    add('training BCE', tr <= train_limit + 1e-8, f'Recomputed training BCE {tr:.6f}; full credit at or below {train_limit}.')
    add('validation BCE', val <= val_limit + 1e-8, f'Recomputed validation BCE {val:.6f}; full credit at or below {val_limit}.')
    add('validation accuracy', accuracy >= .87 - 1e-8,
        f'Validation accuracy at threshold 0.5: {accuracy:.2%}; full credit at or above 87%. Threshold comparison (not separately graded):\n{threshold_feedback}')
    return dict(tests=tests, metrics=metrics)


if __name__ == '__main__':
    part, path, data_path = sys.argv[1:]
    try:
        result = model_result(Path(path), part, json.loads(Path(data_path).read_text()))
    except Exception as error:
        result = dict(tests=[dict(name=f'{part} — evaluation', score=0, max_score=PART_POINTS[part],
                                 output='Unable to evaluate: ' + str(error), visibility='visible')])
    print(json.dumps(result, allow_nan=False))
