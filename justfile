set default-list := true
set dotenv-load := true

alias i := install
install:
	uv sync

alias t := test
test *args:
	just typecheck && echo ''
	uv run pytest {{args}}

typecheck path='src':
	uv run basedpyright {{path}}
