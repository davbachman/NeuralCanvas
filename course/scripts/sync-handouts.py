"""Generate only student handouts for the public app; --check never writes."""
import argparse
import html
from pathlib import Path
import re

COURSE = Path(__file__).resolve().parents[1]
APP = COURSE.parent


def render(source):
    body = source.read_text()
    heading = re.search(r'<h2[^>]*>(.*?)</h2>', body, re.DOTALL)
    if heading is None:
        raise ValueError(f'Missing assignment title: {source}')
    title = html.escape(html.unescape(re.sub(r'<[^>]+>', '', heading[1])))
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} · Neural Canvas</title>
  <style>
    body {{ margin: 0 auto; padding: 1.25rem; max-width: 960px; color: #20242c; background: #fff; font-family: sans-serif; line-height: 1.6; }}
    nav {{ margin-bottom: 1.5rem; }}
    a {{ color: #244c9b; }}
    pre {{ overflow-x: auto; }}
  </style>
</head>
<body>
  <nav aria-label="Course navigation"><a href="../">Open Neural Canvas</a> · <a href="https://github.com/davbachman/NeuralCanvas#course-assignments">All course assignments</a></nav>
  <main>
{body.rstrip()}
  </main>
</body>
</html>
'''


def sync(check=False):
    sources = sorted((COURSE / 'assignments').glob('*/Canvas/description.html'))
    if len(sources) != 19:
        raise ValueError(f'Expected 19 handouts, found {len(sources)}')
    for source in sources:
        target = APP / 'public/course' / (source.parents[1].name + '.html')
        expected = render(source)
        if check:
            if not target.is_file() or target.read_text() != expected:
                raise ValueError(f'Outdated student page: {target}; run {Path(__file__).name}')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(expected)
    print(f'{"Checked" if check else "Updated"} all 19 student pages.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    sync(parser.parse_args().check)
