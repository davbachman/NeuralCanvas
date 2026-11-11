"""Validate matrix model structure and independently recompute MSE/softmax CE."""
import json
from pathlib import Path
import sys
import numpy as np
from graph_eval import graph_from_file, compile_graph, tensor, require, ZERO, FEATURES

VERSION = 'matrices-04-v1'
PART_POINTS = {'matrix_regression': 40, 'iris': 60}


def predictions(polys, x):
    return np.column_stack([sum((v * np.prod(x ** np.array(p[:x.shape[1]]), axis=1)
                                for p, v in poly.terms.items()), np.zeros(len(x))) for poly in polys])


def model_result(path, part, datasets):
    data = datasets[part]
    width, classes = (2, 1) if part == 'matrix_regression' else (4, 3)
    weights = {'saved model and dataset': 5, 'matrix model and scaling': 20 if classes == 1 else 30,
               'training loss': 10, 'validation loss': 5 if classes == 1 else 10}
    if classes == 3:
        weights['validation accuracy'] = 5
    tests, metrics = [], {}
    def add(name, ok, message):
        tests.append(dict(name=f'{part} — {name}', score=weights[name] if ok else 0,
                          max_score=weights[name], output=message, visibility='visible'))
    try:
        graph = graph_from_file(path, data)
        add('saved model and dataset', True, 'Original dataset, labels, and split verified.')
    except Exception as error:
        for name in weights:
            add(name, False, str(error) if name == 'saved model and dataset' else 'Requires a readable saved model and original dataset.')
        return dict(tests=tests, metrics=metrics)
    try:
        polys, params, used, ancestors = compile_graph(graph, width, classes)
        require(used == set(range(width)), 'Use every assigned feature, and no other predictors.')
        basis = {ZERO, *FEATURES[:width]}
        require(all(set(p.terms) <= basis for p in polys), 'Use a linear matrix model without powers or hidden layers.')
        nodes = {n['id']: n for n in graph['nodes']}
        parameter_ids = {key for key, _ in params}
        require(len(parameter_ids) == 2 and len(params) == width*classes+classes,
                'Use exactly two trainable Param blocks: the weight matrix and bias.')
        tensors = {key: tensor(nodes[key]['params'].get('value', .5)) for key in parameter_ids}
        matrix_ids = [key for key, value in tensors.items() if value.shape == (width, classes)]
        require(len(matrix_ids) == 1, f'The weight Param must have shape {width}, {classes}.')
        matrix_id = matrix_ids[0]
        bias_id = next(key for key in parameter_ids if key != matrix_id)
        require(tensors[bias_id].shape == (() if classes == 1 else (classes,)),
                'Use a scalar bias for MPG and a shape-3 bias vector for Iris.')
        active = [nodes[key] for key in ancestors]
        products = [n for n in active if n['type'] == 'matmul']
        require(len(products) == 1 and any(n['type'] == 'concat' for n in active),
                'Assemble feature columns with Concatenate and use one Matrix product in the prediction path.')
        require(any(e['target'] == products[0]['id'] and e.get('inputSlot', 0) == 1 and
                    e['source'] == matrix_id and e.get('sourceSlot', 0) == 0 for e in graph['edges']),
                'Connect the weight matrix directly to the second Matrix product input.')
        x = np.array([r['x'] for r in data['rows']])
        y = np.array([r['y'] for r in data['rows']])
        train = np.array([r['split'] == 'train' for r in data['rows']])
        z = (x-x[train].mean(0))/x[train].std(0)
        # Check every weight/bias direction, not just the submitted numerical predictions.
        for key, index in params:
            original = tensors[key].flat[index].terms.get(ZERO, 0)
            plus = compile_graph(graph, width, classes, (key, index, original+.125))[0]
            minus = compile_graph(graph, width, classes, (key, index, original-.125))[0]
            require(all(set(p.terms) <= basis for p in [*plus, *minus]), 'Parameters must only control linear terms.')
            expected = np.zeros((len(x), classes))
            if key == matrix_id:
                feature, output = divmod(index, classes)
                expected[:, output] = z[:, feature]
            else:
                expected[:, index] = 1
            for delta in [[a-b for a,b in zip(plus,polys)], [a-b for a,b in zip(polys,minus)]]:
                require(np.allclose(predictions(delta,x)/.125, expected, atol=2e-5, rtol=2e-5),
                        'Use columns in the prescribed order, standardize on training rows, and add each bias to its output column.')
        W = np.array([p.terms.get(ZERO, 0) for p in tensors[matrix_id].flat]).reshape(width, classes)
        b = np.array([p.terms.get(ZERO, 0) for p in tensors[bias_id].flat])
        scores = predictions(polys, x)
        require(np.allclose(scores, z @ W + b, atol=1e-5, rtol=1e-5), 'Do not add fixed fitted coefficients or other prediction terms.')
        add('matrix model and scaling', True, 'Correct concatenation, matrix Param, bias, training-fitted scaling, and loss.')
    except Exception as error:
        for name in weights:
            if name != 'saved model and dataset':
                add(name, False, str(error) if name == 'matrix model and scaling' else 'Requires the assigned matrix model.')
        return dict(tests=tests, metrics=metrics)
    if classes == 1:
        losses = (scores[:, 0]-y)**2
        train_limit, validation_limit = 14., 10.
    else:
        # Numerically stable softmax cross-entropy from raw logits, no clipping.
        shifted = scores-scores.max(axis=1, keepdims=True)
        losses = np.log(np.exp(shifted).sum(axis=1))-shifted[np.arange(len(y)), y.astype(int)]
        train_limit, validation_limit = .15, .25
    tr, val = float(losses[train].mean()), float(losses[~train].mean())
    metrics.update(train=tr, validation=val)
    add('training loss', tr <= train_limit+1e-8, f'Recomputed training loss {tr:.6f}; full credit at or below {train_limit}.')
    add('validation loss', val <= validation_limit+1e-8, f'Recomputed validation loss {val:.6f}; full credit at or below {validation_limit}.')
    if classes == 3:
        accuracy = float(np.mean(scores[~train].argmax(axis=1) == y[~train]))
        metrics['accuracy'] = accuracy
        add('validation accuracy', accuracy >= .9-1e-8, f'Validation accuracy {accuracy:.2%}; full credit at or above 90%.')
    return dict(tests=tests, metrics=metrics)


if __name__ == '__main__':
    part, path, data_path = sys.argv[1:]
    try:
        result = model_result(Path(path), part, json.loads(Path(data_path).read_text()))
    except Exception as error:
        result = dict(tests=[dict(name=f'{part} — evaluation', score=0, max_score=PART_POINTS[part],
                                 output='Unable to evaluate: '+str(error), visibility='visible')])
    print(json.dumps(result, allow_nan=False))
