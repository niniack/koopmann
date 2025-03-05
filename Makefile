.PHONY: test
test_all:
	poetry run pytest -s

test_layer:
	poetry run pytest -s tests/test_linear.py tests/test_conv.py

test_block:
	poetry run pytest -s tests/test_linear_res.py tests/test_conv_res.py

test_mlp:
	poetry run pytest -s tests/test_mlp.py

test_resnet:
	poetry run pytest -s tests/test_resnet.py
