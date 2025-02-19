.PHONY: test
test_all:
	poetry run pytest -s

test_mlp:
	poetry run pytest -s tests/test_mlp.py
