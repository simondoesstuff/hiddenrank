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

# Paper populations, one representative iid realization per contextual task.
pilot cores='1':
	env PYTHONPATH='' uv run snakemake --snakefile workflow/Snakefile --configfile config/pilot.yaml --cores {{cores}}

# Expanded contextual ensembles with resumable realization checkpoints.
replication cores='2':
	env PYTHONPATH='' uv run snakemake --snakefile workflow/Snakefile --configfile config/replication.yaml --cores {{cores}}
