# Portrait Map Lab

Generates portrait analysis maps (importance, density, flow, complexity) that drive pen-plotting algorithms in the companion [TypeScript plotter repo](../plotter).

## Setup

Requires Python 3.10+.

```bash
uv sync
```

## Quick Commands

```bash
make serve    # API server → http://127.0.0.1:8100 (docs at /docs)
make test     # run tests
make lint     # ruff check
make fmt      # ruff format
```

## CLI Pipelines

```bash
uv run python scripts/run_pipeline.py <command> path/to/portrait.jpg
```

Commands: `features`, `contour`, `density`, `flow`, `complexity`, `all`.

Use `--help` on any command for options. Common flags:

- `--output-dir` — output directory (default: `output`)
- `--persist` — copy maps to the output directory
- `--export` — generate export bundle for TypeScript (`all` only)

## Development

See `Makefile` for all shortcuts. Full commands:

```bash
uv run pytest              # tests
uv run ruff check src/     # lint
uv run ruff format src/    # format
```

## Further Reading

- [docs/vision.md](docs/vision.md) — full vision and 7-stage roadmap
