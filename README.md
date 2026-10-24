# dev-toolbox

A growing collection of small, well-tested Python command-line utilities.
Pure standard library — no dependencies, works anywhere Python 3.10+ runs.
One new tool at a time, each with tests.

## Usage

```bash
python -m toolbox slugify "Hello World!"        # hello-world
python -m toolbox snake "helloWorld-foo bar"    # hello_world_foo_bar
python -m toolbox camel "hello_world"           # helloWorld
python -m toolbox camel "hello_world" --upper   # HelloWorld
python -m toolbox b64 "hello"                   # aGVsbG8=
python -m toolbox b64 "aGVsbG8=" --decode       # hello
python -m toolbox hash "hello"                  # sha256 hex digest
python -m toolbox hash "hello" --algo md5
python -m toolbox extract_json 'Here: {"a": 1}'  # {"a": 1} (pretty-printed)
```

Or import the functions directly:

```python
from toolbox import texttools
texttools.slugify("My First Post")   # 'my-first-post'
```

## Tools

| Command | What it does |
|---|---|
| `slugify` | Title → URL slug (keeps Unicode letters, e.g. CJK) |
| `snake` | camelCase / PascalCase / kebab-case → snake_case |
| `camel` | snake_case / kebab-case → camelCase or PascalCase |
| `b64` | Base64 encode / decode UTF-8 text |
| `hash` | md5 / sha1 / sha256 hex digest of a text |
| `extract_json` | First valid JSON object/array from LLM output (handles ```json fences and surrounding prose) |

## Roadmap

- `jsonfmt` — validate and pretty-print JSON from a file or stdin
- `urlparse` — break a URL into its components
- `timestamp` — convert between Unix timestamps and human dates
- `password` — generate a strong random password
- `lines` — count lines / words / chars of files (with totals)

## Tests

```bash
python tests/test_texttools.py   # no dependencies
pytest tests/                    # or with pytest
```

## License

MIT
