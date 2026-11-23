Key Paper, by Mastrogiuseppe and Ostojic [Link](https://pubmed.ncbi.nlm.nih.gov/30057201/)
Provided locally, [docs/ostojic.pdf](docs/ostojic.pdf)

Please maintain [docs/](docs/) as you add content. Even few sentence descriptions are fine.

Feel free to add new top-level packages in src/ or add to existing ones.

Key **just recipes**

- `just i` to install/setup
- `just t` to run [tests](tests/)
- `just typecheck` to run code linter & typecheck (code quality check)

## Setup

**Setup**: `just i` or `uv sync` (if you don't have just)  
Run modules: `uv run src/example`
Run files: `uv run src/example/__main__.py`

This project uses:

1. nix: setup environment, including python itself
2. uv: this packages python and fetches dependencies instead of pypi
3. snakemake: this manages data/plots to enforce reproducibility; declarative artifacts
4. just: this lets you write just scripts which contain convenience commands/scripts

You **do not** need to have any of these to run code. Individual python files can be run directly if necessary.
`uv` is recommended for _stability_.  
`snakemake` is recommended for _reproducibility_.
`nix` is not recommended if you haven't seen it before and may be complicated to setup, but it can be used to install/setup everything for you automatically.
