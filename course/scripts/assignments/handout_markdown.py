"""Render the course handouts as GitHub Markdown using only the standard library."""
from dataclasses import dataclass, field
import html
from html.parser import HTMLParser
import re


@dataclass
class Element:
    tag: str
    attrs: dict = field(default_factory=dict)
    children: list = field(default_factory=list)


class Handout(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=True)
        self.root = Element('div')
        self.stack = [self.root]
        self.datasets = []
        self.feed(source)
        self.close()
        if len(self.stack) != 1:
            raise ValueError('Unclosed handout element')

    def handle_starttag(self, tag, attrs):
        node = Element(tag, dict(attrs))
        self.stack[-1].children.append(node)
        if 'data-library-file' in node.attrs:
            self.datasets.append(node.attrs['data-library-file'])
        if tag != 'br':
            self.stack.append(node)

    def handle_endtag(self, tag):
        if tag == 'br':
            return
        if self.stack[-1].tag != tag:
            raise ValueError(f'Mismatched handout element: {tag}')
        self.stack.pop()

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def plain(node):
    return node if isinstance(node, str) else ''.join(plain(child) for child in node.children)


def children(node):
    return ''.join(render(child) for child in node.children)


def inline(node):
    return re.sub(r'\s+', ' ', children(node)).strip()


def render(node):
    if isinstance(node, str):
        text = re.sub(r'\s+', ' ', node)
        return re.sub(r'([\\`*_\[\]])', r'\\\1', html.escape(text, quote=False))
    tag = node.tag
    if tag in ('strong', 'b', 'em', 'i'):
        marker = '**' if tag in ('strong', 'b') else '*'
        return marker + inline(node) + marker
    if tag == 'code':
        value = plain(node)
        fence = '`' * (max((len(s) for s in re.findall(r'`+', value)), default=0) + 1)
        return fence + (' ' if '`' in value else '') + value + (' ' if '`' in value else '') + fence
    if tag in ('sub', 'sup'):
        return f'<{tag}>' + inline(node) + f'</{tag}>'
    if tag == 'a':
        return '[' + inline(node) + '](' + node.attrs['href'] + ')'
    if tag == 'br':
        return '<br>'
    if tag in ('h2', 'h3'):
        return '\n\n' + '#' * (int(tag[1]) - 1) + ' ' + inline(node) + '\n\n'
    if tag == 'p':
        return '\n\n' + inline(node) + '\n\n'
    if tag in ('div', 'small', 'thead', 'tbody'):
        return children(node)
    if tag in ('ol', 'ul'):
        items = [child for child in node.children if isinstance(child, Element) and child.tag == 'li']
        lines = []
        for i, item in enumerate(items, 1):
            marker = f'{i}. ' if tag == 'ol' else '- '
            body = children(item).strip().splitlines()
            lines.append(marker + ('\n' + ' ' * len(marker)).join(body))
        return '\n\n' + '\n'.join(lines) + '\n\n'
    if tag == 'blockquote':
        return '\n\n' + '\n'.join('> ' + line for line in children(node).strip().splitlines()) + '\n\n'
    if tag == 'pre':
        return '\n\n```text\n' + plain(node).strip() + '\n```\n\n'
    if tag == 'details':
        return '\n\n<details>\n' + children(node).strip() + '\n\n</details>\n\n'
    if tag == 'summary':
        return '<summary>' + inline(node) + '</summary>\n\n'
    if tag == 'table':
        rows = []
        def collect(parent):
            for child in parent.children:
                if isinstance(child, Element):
                    if child.tag == 'tr':
                        row = []
                        for cell in child.children:
                            if isinstance(cell, Element) and cell.tag in ('td', 'th'):
                                row.append(inline(cell).replace('|', '\\|'))
                                row.extend([''] * (int(cell.attrs.get('colspan', 1)) - 1))
                        rows.append(row)
                    else:
                        collect(child)
        collect(node)
        width = max(map(len, rows))
        rows = [row + [''] * (width - len(row)) for row in rows]
        rows.insert(1, ['---'] * width)
        captions = [child for child in node.children if isinstance(child, Element) and child.tag == 'caption']
        caption = '**' + inline(captions[0]) + '**\n\n' if captions else ''
        return '\n\n' + caption + '\n'.join('| ' + ' | '.join(row) + ' |' for row in rows) + '\n\n'
    raise ValueError(f'Unsupported handout element: {tag}')


def markdown(source):
    body = children(Handout(source).root)
    body = re.sub(r'\n[ \t]+\n', '\n\n', body)
    body = re.sub(r'\n{3,}', '\n\n', body).strip()
    return '<!-- Generated from description.html; run course/scripts/sync-handouts.py after editing. -->\n\n' + body + '\n'
