"""Restricted graph evaluator for Assignment 3; never execute student code."""
import ast
import json
import math
import os
from pathlib import Path
import sys
import numpy as np

VERSION = 'classification-03-v1'
ZERO = (0, 0, 0)
X, Z, Y = (1, 0, 0), (0, 1, 0), (0, 0, 1)
ALLOWED = {'dataset', 'input', 'target', 'weight', 'bias', 'arithmetic',
           'add', 'multiply', 'matmul', 'concat', 'reshape', 'transpose',
           'tensor-transform', 'loss', 'standardize', 'activation'}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and abs(value) <= 1e9

class Poly:
    def __init__(self, terms):
        require(all(finite(v) for v in terms.values()), 'Nonfinite or excessive calculation.')
        require(all(sum(k) <= 6 for k in terms), 'Use only the assigned linear features.')
        self.terms = {k: float(v) for k, v in terms.items() if v != 0}

    @staticmethod
    def of(value):
        return value if isinstance(value, Poly) else Poly({ZERO: value})

    def __add__(self, other):
        other = Poly.of(other)
        terms = dict(self.terms)
        for k, v in other.terms.items():
            terms[k] = terms.get(k, 0) + v
        return Poly(terms)

    __radd__ = __add__

    def __neg__(self):
        return Poly({k: -v for k, v in self.terms.items()})

    def __sub__(self, other):
        return self + -Poly.of(other)

    def __rsub__(self, other):
        return Poly.of(other) + -self

    def __mul__(self, other):
        terms = {}
        for k, v in self.terms.items():
            for j, w in Poly.of(other).terms.items():
                power = tuple(a+b for a, b in zip(k, j))
                terms[power] = terms.get(power, 0) + v*w
        return Poly(terms)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = Poly.of(other)
        require(set(other.terms) <= {ZERO} and other.terms.get(ZERO, 0) != 0,
                'Divide only by a nonzero constant.')
        return self * (1 / other.terms[ZERO])

    def __pow__(self, other):
        other = Poly.of(other)
        exponent = other.terms.get(ZERO, 0)
        require(set(other.terms) <= {ZERO} and exponent in range(6),
                'Use a constant integer exponent from 0 through 5.')
        result = Poly.of(1)
        for _ in range(int(exponent)):
            result = result * self
        return result

def tensor(value):
    if finite(value):
        return np.array(Poly.of(value), dtype=object)
    require(isinstance(value, dict), 'A parameter or constant needs a numeric value.')
    shape, data = value.get('shape'), value.get('data')
    require(isinstance(shape, list) and len(shape) <= 3 and
            all(type(d) is int and 1 <= d <= 64 for d in shape),
            'Use scalar or small tensor parameters (at most 64 values).')
    require(isinstance(data, list) and len(data) == math.prod(shape) <= 64 and
            all(finite(v) for v in data), 'Invalid parameter tensor.')
    return np.array([Poly.of(v) for v in data], dtype=object).reshape(shape)

def expression(text, args):
    require(isinstance(text, str) and len(text) <= 256, 'Arithmetic expression is too long.')
    tree = ast.parse(text.replace('^', '**'), mode='eval')
    require(sum(1 for _ in ast.walk(tree)) <= 100, 'Arithmetic expression is too complex.')
    used = set()
    def walk(n):
        if isinstance(n, ast.Constant) and finite(n.value):
            return tensor(n.value)
        if isinstance(n, ast.Name) and n.id.startswith('x') and n.id[1:].isdigit():
            index = int(n.id[1:])-1
            require(0 <= index < len(args), 'Arithmetic input is missing.')
            used.add(index)
            return args[index]
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            return np.asarray(walk(n.operand) if isinstance(n.op, ast.UAdd) else -walk(n.operand), dtype=object)
        if isinstance(n, ast.BinOp):
            a, b = walk(n.left), walk(n.right)
            if isinstance(n.op, ast.Add): return np.asarray(a+b, dtype=object)
            if isinstance(n.op, ast.Sub): return np.asarray(a-b, dtype=object)
            if isinstance(n.op, ast.Mult): return np.asarray(a*b, dtype=object)
            if isinstance(n.op, ast.Div): return np.asarray(a/b, dtype=object)
            if isinstance(n.op, ast.Pow): return np.asarray(a**b, dtype=object)
        raise ValueError('Use only x1, x2, …, numbers, parentheses, and + - * / ^.')
    result = walk(tree.body)
    require(used == set(range(len(args))), 'Connect exactly the arithmetic inputs used.')
    return result

def read_json(path):
    require(path.is_file() and not path.is_symlink(), 'Required submission file is missing.')
    require(path.stat().st_size <= 4_000_000, 'Each JSON must be at most 4 MB.')
    return json.loads(path.read_text(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON.')))

def graph_from_file(path, data):
    file = read_json(path)
    require(isinstance(file, dict) and file.get('kind') in ('neural-canvas-state', 'backprop-builder-state') and file.get('version') == 1,
            'Upload a JSON downloaded using File → Save in Neural Canvas.')
    graph = file.get('state', {}).get('graph')
    require(isinstance(graph, dict), 'The file has no saved graph.')
    nodes, edges = graph.get('nodes'), graph.get('edges')
    require(isinstance(nodes, list) and 1 <= len(nodes) <= 100 and
            isinstance(edges, list) and len(edges) <= 200, 'Limit models to 100 blocks and 200 wires.')
    require(all(isinstance(n, dict) and isinstance(n.get('id'), str) and len(n['id']) <= 128 and
                isinstance(n.get('params'), dict) for n in nodes), 'Invalid blocks.')
    ids = [n['id'] for n in nodes]
    require(len(set(ids)) == len(ids), 'Block IDs must be unique.')
    sources = [n for n in nodes if n.get('type') == 'dataset']
    require(len(sources) == 1, 'Use exactly one Dataset block.')
    csv = sources[0]['params'].get('customCsv', {})
    require(sources[0]['params'].get('dataset') == 'custom-csv' and isinstance(csv, dict) and
            csv.get('hasHeader') is True and csv.get('targetColumn') == 2 and
            csv.get('task') == 'binary-classification', 'Import breast-cancer.csv, with malignant as the binary-classification target.')
    expected = data['rows']
    rows = csv.get('rows', [])
    require(isinstance(rows, list) and len(rows) == len(expected)+1 and
            rows[0] == ['mean_radius', 'mean_texture', 'malignant'] and
            csv.get('splits') == [r['split'] for r in expected],
            'Use the provided breast-cancer.csv without changing its columns, rows, or split.')
    for actual, canonical in zip(rows[1:], expected):
        require(isinstance(actual, list) and len(actual) == 3 and
                all(isinstance(v, str) and len(v) <= 64 for v in actual), 'Invalid CSV cells.')
        require(all(abs(float(v)-canonical[k]) <= 1e-9 for v, k in zip(actual, ['radius', 'texture', 'malignant'])),
                'The saved dataset differs from the assignment CSV.')
    for edge in edges:
        require(isinstance(edge, dict) and edge.get('source') in ids and edge.get('target') in ids,
                'A wire refers to an unknown block.')
        require(all(type(edge.get(k, 0)) is int and 0 <= edge.get(k, 0) < 64
                    for k in ['sourceSlot', 'inputSlot']), 'Invalid wire ports.')
    return graph

def compile_graph(graph, override=None):
    nodes = {n['id']: n for n in graph['nodes']}
    losses = [n for n in nodes.values() if n.get('type') == 'loss']
    require(len(losses) == 1, 'Use one binary-cross-entropy Loss block.')
    loss = losses[0]
    logits_loss = loss['params'].get('loss') == 'binary-cross-entropy-with-logits'
    require(logits_loss or loss['params'].get('loss') == 'binary-cross-entropy', 'Select Binary cross entropy (logits).')
    incoming = {key: [] for key in nodes}
    for e in graph['edges']:
        incoming[e['target']].append(e)
    for edges in incoming.values():
        edges.sort(key=lambda e: e.get('inputSlot', 0))
        require([e.get('inputSlot', 0) for e in edges] == list(range(len(edges))),
                'Connect each input port exactly once, without gaps.')
    prediction_edges = incoming[loss['id']]
    require(len(prediction_edges) == 2, 'Connect the prediction and malignant target to Loss.')
    sigmoid_id = None if logits_loss else prediction_edges[0]['source']
    if not logits_loss:
        require(nodes[sigmoid_id].get('type') == 'activation' and
                nodes[sigmoid_id]['params'].get('activation') == 'sigmoid',
                'Legacy probability BCE requires Sigmoid directly into Loss; for raw scores select Binary cross entropy (logits).')
    memo, visiting, trainable, feature_slots = {}, set(), [], set()
    has_parameter, scaled_features = {}, set()

    def value(key):
        if key in memo: return memo[key]
        require(key not in visiting, 'The model contains a cycle.')
        visiting.add(key)
        node, edges = nodes[key], incoming[key]
        kind, p = node.get('type'), node['params']
        require(kind in ALLOWED, 'Use linear arithmetic and feature scaling; sigmoid is internal to the logits loss.')
        if kind == 'dataset':
            require(not edges, 'Dataset blocks cannot have inputs.')
            out = [np.array(Poly({power: 1}), dtype=object) for power in [X, Z, Y]]
        else:
            args = []
            for edge in edges:
                source = nodes[edge['source']]
                slot = edge.get('sourceSlot', 0)
                source_value = value(edge['source'])
                if source['type'] == 'dataset':
                    require(slot < 3, 'Invalid dataset column.')
                    args.append(source_value[slot])
                else:
                    require(slot == 0, 'Use the first output port.')
                    args.append(source_value)
            if kind == 'tensor-transform':
                kind = p.get('transform', 'reshape')
            if kind in ('input', 'target'):
                require(len(args) <= 1, 'Input/Target blocks accept at most one wire.')
                out = args[0] if args else tensor(p.get('value', 1 if kind == 'input' else 0))
            elif kind in ('weight', 'bias'):
                require(not args, 'Param blocks must not have incoming wires.')
                out = tensor(p.get('value', .5))
                for i in range(out.size):
                    trainable.append((key, i))
                    if override and override[:2] == (key, i):
                        out.flat[i] = Poly.of(override[2])
            elif kind == 'arithmetic':
                out = expression(p.get('expression', 'x1 * x2'), args)
            elif kind == 'standardize':
                require(len(args) == 1, 'Standardize features needs one input.')
                stats = p.get('standardization')
                require(isinstance(stats, dict), 'Fit Standardize features on training rows first.')
                mean, scale = stats.get('mean'), stats.get('scale')
                require(isinstance(mean, list) and isinstance(scale, list) and 0 < len(mean) == len(scale) <= 64 and
                        all(finite(v) for v in mean) and all(finite(v) and v > 0 for v in scale) and
                        type(stats.get('count')) is int and stats['count'] > 0, 'Invalid fitted standardization statistics.')
                require(len(mean) == 1 or (args[0].ndim > 0 and args[0].shape[-1] == len(mean)), 'Standardization feature count changed. Refit.')
                out = np.array([(cell - mean[i % len(mean)]) / scale[i % len(scale)] for i, cell in enumerate(args[0].flat)], dtype=object).reshape(args[0].shape)
            elif kind in ('add', 'multiply'):
                require(len(args) >= 2, 'Connect at least two inputs.')
                out = args[0]
                for arg in args[1:]:
                    out = out+arg if kind == 'add' else out*arg
            elif kind == 'matmul':
                require(len(args) == 2, 'Matrix product needs two inputs.')
                out = args[0] @ args[1]
            elif kind == 'concat':
                require(len(args) >= 2, 'Concatenate needs at least two inputs.')
                axis = p.get('axis', 1)
                require(type(axis) is int and 0 <= axis < 3, 'Invalid concatenation axis.')
                out = np.concatenate([a.reshape(-1, 1) if axis == 1 and a.ndim == 1 else a for a in args], axis=axis)
            elif kind == 'reshape':
                require(len(args) == 1, 'Reshape needs one input.')
                shape = p.get('shape', list(args[0].shape))
                require(isinstance(shape, list) and len(shape) <= 3 and
                        all(type(d) is int and (d == -1 or 1 <= d <= 64) for d in shape), 'Invalid reshape.')
                out = args[0].reshape(shape)
            elif kind == 'transpose':
                require(len(args) == 1, 'Transpose needs one input.')
                out = np.transpose(args[0], p.get('axes'))
            elif kind == 'activation':
                require(key == sigmoid_id and p.get('activation') == 'sigmoid' and len(args) == 1,
                        'Connect the linear score directly to logits Loss. Only legacy probability BCE takes a sigmoid input.')
                out = args[0]  # Compile the logit symbolically; apply sigmoid independently when scoring.
            elif kind == 'loss':
                require(len(args) == 2 and args[0].size == args[1].size == 1, 'Predict one value per example and connect the target.')
                require(args[1].flat[0].terms == {Y: 1}, 'The second Loss input must be the malignant target.')
                out = args[0]
            else:
                raise ValueError('This tensor transformation is not needed for this assignment.')
        if kind != 'dataset':
            out = np.asarray(out, dtype=object)
            require(isinstance(out, np.ndarray) and out.size <= 64 and out.ndim <= 3, 'Keep tensors at most 64 values.')
        has_parameter[key] = kind in ('weight', 'bias') or any(has_parameter[e['source']] for e in edges)
        if kind not in ('dataset', 'loss') and not has_parameter[key]:
            for cell in out.flat:
                for slot, power in enumerate([X, Z]):
                    if set(cell.terms) <= {ZERO, power} and 0 < abs(cell.terms.get(power, 0)) < 1:
                        scaled_features.add((key, slot))
        memo[key] = out
        visiting.remove(key)
        return out

    result = value(loss['id']).flat[0]
    # Target leakage is rejected even if algebraic cancellation hides its effect.
    prediction_edge = incoming[loss['id']][0]
    seen = set()
    def inspect(edge):
        source, slot = nodes[edge['source']], edge.get('sourceSlot', 0)
        if source['type'] == 'dataset':
            feature_slots.add(slot)
        elif source['id'] not in seen:
            seen.add(source['id'])
            for previous in incoming[source['id']]: inspect(previous)
    inspect(prediction_edge)
    require(2 not in feature_slots, 'Do not feed the target into your prediction.')
    return result, trainable, feature_slots, {slot for key, slot in scaled_features if key in seen}
