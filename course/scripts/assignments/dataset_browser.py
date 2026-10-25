"""Canonical classroom choices in Neural Canvas's categorized dataset browser."""
from html import escape

CHOICES = {
    'mpg.csv': dict(id='auto-mpg', category='Regression', family='Auto MPG', configuration='Full sample'),
    'mpg-regularization.csv': dict(id='auto-mpg-small', category='Regression', family='Auto MPG', configuration='Small training sample'),
    'breast-cancer.csv': dict(id='breast-cancer', category='Classification', family='Breast cancer measurements'),
    'iris.csv': dict(id='iris', category='Classification', family='Iris flowers'),
    'concrete.csv': dict(id='concrete', category='Regression', family='Concrete strength'),
    'rings.csv': dict(id='rings', category='Classification', family='Concentric rings'),
    'imdb.json': dict(id='imdb', category='Text and sequences', family='Movie reviews'),
    'football-counts.json': dict(id='football-counts', category='Text and sequences', family='Football stories', preset='Football location · word counts'),
    'football-tokens.json': dict(id='football-tokens', category='Text and sequences', family='Football stories', preset='Football location · word counts', control='Model inputs and prediction', value='One ordered token sequence → answer class'),
    'stories.json': dict(id='football-mixed', category='Text and sequences', family='Football stories', preset='Mixed-question classifier'),
    'football-test.json': dict(id='football-benchmark', category='Text and sequences', family='Football stories', preset='Mixed training · football benchmark'),
    'football-narratives-last.json': dict(id='narrative-last', category='Text and sequences', family='Football stories', preset='Narrative generation', control='Training targets', value='Final position only'),
    'football-narratives-all.json': dict(id='narrative-all', category='Text and sequences', family='Football stories', preset='Narrative generation', control='Training targets', value='Every input position'),
    'story-questions.json': dict(id='football-qa', category='Text and sequences', family='Football stories', preset='Answer-token fine-tuning'),
    'digits-8x8': dict(id='digits-8x8', category='Images', family='Handwritten digits'),
}


def dataset_choice(filename, *, context=None):
    choice = CHOICES[filename]
    strong = lambda value: '<strong>' + escape(value) + '</strong>'
    text = (escape(context) + ' ' if context else '')
    text += ('Select your Dataset block and open ' + strong('Details → Browse datasets…') + '. Choose '
             + strong(choice['category']) + ' → ' + strong(choice['family']) + '.')
    if choice.get('configuration'):
        text += ' Set ' + strong('Configuration') + ' to ' + strong(choice['configuration']) + '.'
    if choice.get('preset'):
        text += ' Choose the ' + strong(choice['preset']) + ' preset.'
    if choice.get('control'):
        text += ' Set ' + strong(choice['control']) + ' to ' + strong(choice['value']) + '.'
    text += ' Click ' + strong('Use this dataset') + '. Keep the supplied data and split unchanged.'
    return (f'<p data-library-preset="{choice["id"]}" data-library-file="{filename}">' + text + '</p>')


def dataset_choices(filename=None, *, columns=None, context=None):
    if columns is not None:
        return ''.join(dataset_choice(name, context=f'For {label[0].lower() + label[1:] if not label.startswith("Part ") else label}:')
                       for label, name in columns.items())
    return dataset_choice(filename, context=context)
