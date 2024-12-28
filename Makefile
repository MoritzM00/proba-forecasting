setup: initialize_git install

initialize_git:
	git init

install:
	uv sync --all-extras --dev
	uv run pre-commit install --hook-type pre-push --hook-type post-checkout --hook-type pre-commit
	uv pip install -e .

test:
	uv run pytest

docs_view:
	uv run pdoc probafcst --docformat numpy --mermaid --math

docs_save:
	uv run pdoc probafcst -o docs --docformat numpy --mermaid --math

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache

submit:
	@rm probafcst_cache.sqlite
	@rm .cache.sqlite
	@dvc repro -f
