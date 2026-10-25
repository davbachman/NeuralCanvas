"""Shared presentation of current handout settings and assignment handoffs."""


def table(caption, headers, rows, *, panel=None):
    marker = f' data-settings-panel="{panel}"' if panel else ''
    return ('<div style="overflow-x:auto"><table border="1" cellpadding="6" '
            'style="border-collapse:collapse;width:100%"' + marker + '><caption>' + caption + '</caption><thead><tr>' +
            ''.join('<th scope="col">' + h + '</th>' for h in headers) + '</tr></thead><tbody>' +
            ''.join('<tr>' + ''.join('<td>' + str(v) + '</td>' for v in row) + '</tr>' for row in rows) +
            '</tbody></table></div>')


def training(day):
    epochs = '10' if day in (8, 16, 17, 18) else '5'
    labels = ['Epochs per run', 'Report loss every', 'Examples per update',
              'Reshuffle training examples each epoch', 'Exact learning rate',
              'Training execution', 'Backend', 'Optimizer',
              'Gradient norm limit (0 = off)', 'Early stopping patience (0 = off)']
    values = [epochs, '5' if day in (8, 9, 10, 19) else '1', '32', 'Checked',
              '0.02' if day == 8 else '0.003', 'Tensor engine · minibatches',
              'WebGL' if day == 19 else 'Tensor CPU', 'Adam', '0' if day == 8 else '1', '0']
    headers = ['Control', 'Value']
    rows = list(zip(labels, values))
    if day == 7:
        reviews = ['5', '1', '32', 'Checked', '0.01', 'Tensor engine · minibatches', 'Tensor CPU', 'Adam', '0', '0']
        football = ['100', '10', '32', 'Checked', '0.02', 'Tensor engine · minibatches', 'Tensor CPU', 'Adam', '0', '0']
        headers = ['Control', 'Movie reviews', 'Football questions']
        rows = list(zip(labels, reviews, football))
    return ('<p>First set <strong>Training execution</strong> to <strong>Tensor engine · minibatches</strong> '
            'to show the Tensor controls. Then follow the Train tab from top to bottom:</p>' +
            table('Training settings — Train tab', headers, rows, panel='train') +
            table('Loss block → Details → Parameter regularization', ['Control', 'Value'],
                  [('Penalty', 'None')], panel='regularization'))


def initialization(day):
    tables = 'Word embedding table' if day in (8, 11, 12, 13) else 'Word and position tables'
    rows = [(tables, 'uniform'),
            ('Other matrices', 'Xavier · matrices'), ('Bias vectors', 'zeros')]
    if day == 7:
        rows = [('Weight matrix', 'Xavier · matrices'), ('Bias vector', 'zeros')]
    if day >= 16:
        rows.extend([('Layer norm scales', 'ones'), ('Layer norm offsets', 'zeros')])
    return table('Param → Initialize choices', ['Param block', 'Initialize'], rows, panel='initializer')


# Keep the old helper name for existing release builders. It now selects the
# corresponding library configuration rather than opening a file importer.
from dataset_browser import dataset_choices as prepared_import


def dataset_properties(day, part=None):
    """Describe supplied data rather than pretending these are import controls."""
    headers = ['Dataset property', 'Supplied value']
    if day == 7:
        reviews = part == 'reviews'
        rows = [('Vocabulary entries', '1000' if reviews else '16'),
                ('Text tokens used per example', 'Up to 128' if reviews else '23'),
                ('Word-count output shape', '1 × 1000' if reviews else '1 × 16'),
                ('Answer classes', '2: negative and positive' if reviews else '6 rooms'),
                ('Training examples', '1200' if reviews else '480'),
                ('Held-out examples', '400' if reviews else '240')]
    elif day == 8:
        rows = [('Vocabulary entries', '17, including unknown and padding'),
                ('Token IDs per example', '23'), ('Answer classes', '6 rooms'),
                ('Training examples', '480'), ('Held-out examples', '240')]
    elif day in (9, 10):
        rows = [('Vocabulary entries', '25, including unknown and padding'),
                ('story token IDs', '32'), ('question token IDs', '8'),
                ('story word mask', '32 × 1'), ('question word mask', '8 × 1'),
                ('answer', '1 class index; 9 answer classes'),
                ('Training examples in either QA preset', '3840'),
                ('Held-out examples — Mixed-question classifier', '1920'),
                ('Held-out examples — Mixed training · football benchmark', '240')]
    elif day == 19:
        rows = [('Vocabulary entries', '25, including unknown and padding'),
                ('Input token IDs', '23: story followed by question'),
                ('Next-token target IDs', '1: the answer, excluded from the input'),
                ('Training questions', '3840'), ('Held-out football questions', '240')]
    else:
        rows = [('Vocabulary entries', '25, including unknown and padding'),
                ('Input token IDs', '32'), ('Position IDs', '32'),
                ('Next-token target IDs', '1' if day <= 15 else '32'),
                ('Training windows', '1534'), ('Held-out windows', '765')]
        if day == 16:
            headers = ['Dataset property', 'Parts A and C: last', 'Part B: all']
            rows = [(name, '1', '32') if name == 'Next-token target IDs' else (name, value, value)
                    for name, value in rows]
    return table('Prepared dataset properties', headers, rows)


def generation():
    return table('Optional generation controls', ['Control', 'Value'], [
        ('Prompt and generated text', 'Your chosen prompt; restore it before each comparison'),
        ('Decoding', 'Sample'), ('Temperature', 'The value being explored'), ('Top-k', '25')], panel='text-generation')


def optimizer_experiments():
    rows = [('Epochs per run', '10'), ('Report loss every', '1'),
            ('Exact learning rate', '0.003')]
    return (table('Optional Adam comparison — changed Train controls', ['Control', 'Value'],
                  [*rows, ('Optimizer', 'Adam')], panel='train') +
            table('Optional AdamW experiment — changed Train controls', ['Control', 'Value'],
                  [*rows, ('Optimizer', 'AdamW'), ('Weight decay', '0.1')], panel='train'))


def optimizer_variations():
    return table('Optional AdamW variations — changed Train controls',
                 ['Control', 'Smaller decay', 'Smaller learning rate'], [
                     ('Exact learning rate', '0.003', '0.001'), ('Weight decay', '0.01', '0.1')], panel='train')


HANDOFF = {
    8: ('Assignment 7’s <strong>stories.json model</strong> in a copy',
        'Keep the Dataset and classification loss/probability pattern; switch counts to ordered token IDs and add an MLP. Then copy Part A to build Part B.',
        '<strong>onehot.json</strong> and <strong>embedding.json</strong>; keep embedding.json for Assignment 9.'),
    11: ('A new graph; keep Assignment 10’s <strong>attention.json</strong> and <strong>football.json</strong>',
         'Begin next-token generation with a deliberately small bigram model, using familiar embedding, matrix/bias, loss and Softmax operations.',
         '<strong>bigram.json</strong>; keep it for Assignment 12. Keep Assignment 10’s football.json for the capstone.'),
    13: ('Assignment 12’s <strong>base.json</strong> (the unchanged bigram model)',
         'Keep the Dataset, word embeddings, loss and Softmax. Replace the final-token input path with a flattened context and hidden MLP; resize the output projection.',
         '<strong>context.json</strong>; use a copy in Assignment 14.'),
    14: ('A copy of Assignment 13’s <strong>context.json</strong>; keep Assignment 10’s attention.json nearby as a reference',
         'Keep the Dataset, token embeddings, loss and output Softmax. Replace the flattened context MLP with self-attention, add learned positions and a causal mask, and resize the vocabulary projection.',
         '<strong>self-attention.json</strong>; use a copy in Assignment 15.'),
    15: ('A copy of Assignment 14’s <strong>self-attention.json</strong>',
         'Keep embeddings, self-attention, causal masking and the vocabulary output. Add the attention output projection, two residual routes and a shared MLP.',
         '<strong>residual.json</strong>; use a copy in Assignment 16.'),
    16: ('A copy of Assignment 15’s <strong>residual.json</strong>',
         'Part A adds normalization. Copy A for independently trained all-position Part B. Copy trained B for fixed-weight final-position evaluation in C.',
         '<strong>transformer.json</strong>, <strong>all_positions.json</strong>, <strong>last_position.json</strong>; keep all_positions.json for Assignment 17.'),
    17: ('Assignment 16 Part B’s <strong>all_positions.json</strong>',
         'Save the trained model as one_block.json. In a separate copy, duplicate and reconnect the transformer block with independent Params, then randomize the two-block model.',
         '<strong>one_block.json</strong> and <strong>stacked.json</strong>; keep stacked.json for Assignment 18.'),
    18: ('A copy of Assignment 17’s <strong>stacked.json</strong>',
         'Replace each block’s single attention head with two independent heads while keeping the total feature width at 16. Randomize the modified architecture.',
         '<strong>multihead.json</strong>; keep this trained generator for Assignment 19.'),
    12: ('Assignment 11’s trained <strong>bigram.json</strong>',
         'Save base.json, then add temperature branches in a copy. Preserve every learned parameter; no new training.',
         '<strong>base.json</strong> and <strong>sampling.json</strong>. Use base.json to build the context model in Assignment 13.'),
    19: ('Assignment 18’s trained <strong>multihead.json</strong> and Assignment 10’s <strong>football.json</strong>',
         'Preserve the generator’s weights, select the QA dataset in a copy, and change the output Slice to predict only the answer. Fine-tune a copy of that baseline.',
         '<strong>earlier.json</strong>, <strong>pretrained.json</strong>, <strong>before.json</strong>, <strong>after.json</strong>.'),
}


def handoff(day):
    start, change, save = HANDOFF[day]
    return table('Your starting point and handoff', ['Step', 'What to use'],
                 [('Start with', start), ('Modify', change), ('Save and keep', save)])


def prepared_data(day):
    if day == 16:
        return prepared_import(columns={'Part A and evaluation copy C': 'football-narratives-last.json',
                                        'Part B': 'football-narratives-all.json'}) + dataset_properties(day)
    if day == 19:
        return prepared_import('story-questions.json', context='For the QA copy in Part A (after saving pretrained.json):') + dataset_properties(day)
    return prepared_import('football-narratives-last.json' if day <= 15 else 'football-narratives-all.json') + dataset_properties(day)


def organize_narrative(html, day):
    """Format the existing generator output without changing model/rubric rules."""
    import re
    if '<caption>Your starting point and handoff</caption>' not in html:
        end = html.index('</h2>') + len('</h2>')
        html = html[:end] + '\n' + handoff(day) + html[end:]
    if '<caption>Prepared dataset' not in html:
        # Keep the existing corpus explanation; put the operational settings next
        # to it, before students start editing the architecture.
        needle = '<h3>'
        i = html.find(needle, html.index('</h2>'))
        html = html[:i] + prepared_data(day) + html[i:]
    if day != 12:
        html, n = re.subn(r'(<h3>Train and save</h3>)<p>Use .*?</p>\s*<p><strong>Backend:</strong>.*?</p>\s*<p>Set each Param.*?</p>',
                         lambda m: m[1] + training(day) + '<p>Tensor CPU is the validated backend. Auto or WebGL may be used; speed depends on the computer and browser. Use the same backend for timed comparisons.</p>' + initialization(day) + '<p>Apply changed shapes, check the saved Initialize choices, then use <strong>Train → Randomize parameters</strong>. Remove unused Params after an architecture change. Each changed architecture starts fresh; evaluation copies retain their trained values.</p>',
                         html, count=1, flags=re.S)
        if n != 1 and '<caption>Training settings' not in html:
            raise ValueError(f'Cannot locate Assignment {day} training instructions.')
    if day == 12:
        start = html.index('<h3>Optional generation</h3>')
        end = html.index('<!-- BEGIN PARAMETER HINT', start)
        html = html[:start] + '<h3>Optional generation</h3><p>In <strong>Dataset → Details → Continue the text</strong>, use the same prompt for each comparison:</p>' + generation() + '<p>Choose <strong>Sample</strong> under <strong>Decoding</strong> to show Temperature and Top-k. Compare separately with <strong>Most likely</strong>. These controls govern generation; the extra canvas branches let you inspect the mathematics and do not set the generation controls. The same generation controls work with every later generative model. Compare variety and repetition; the bigram model still sees only the last token, so sampling cannot give it a longer memory. The raw Softmax branches include padding; generation excludes padding and renormalizes the remaining probabilities. The required artifacts are base.json and sampling.json; no saved file proves which generation buttons you clicked.</p>' + html[end:]
    # Layer norm appears first in 16; avoid a premature initializer reference.
    if day < 16:
        html = html.replace(', and <strong>Ones/Zeros</strong> for Layer norm scales/offsets when present', '')
    return html
