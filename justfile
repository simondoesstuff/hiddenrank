set default-list := true
set dotenv-load := true

alias i := install
install:
	uv sync

alias t := test
test *args:
	just typecheck && echo ''
	env PYTHONPATH='' uv run pytest {{args}}

typecheck path='src':
	env PYTHONPATH='' uv run basedpyright {{path}}

# Small seeded figure recreations (PNG + SVG).
figures cores='1':
	env PYTHONPATH='' uv run snakemake --snakefile workflow/Snakefile --cores {{cores}}

# Original sizes and ensemble counts; run on a suitable compute host.
paper cores='1':
	env PYTHONPATH='' uv run snakemake --snakefile workflow/Snakefile --configfile config/paper.yaml --cores {{cores}}

dry-run:
	env PYTHONPATH='' uv run snakemake --snakefile workflow/Snakefile --dry-run
