.PHONY: test test_all test_components test_models \
        test_layer test_block test_mlp test_resnet \
        test_autoencoder test_datasets test_preprocessing

test_all:
	uv run pytest -s

###
test_components:
	uv run pytest -s tests/test_linear.py tests/test_conv.py tests/test_linear_block.py tests/test_conv_block.py

test_models:
	uv run pytest -s tests/test_mlp.py tests/test_resnet.py tests/test_resmlp.py tests/test_autoencoder.py

###
test_layer:
	uv run pytest -s tests/test_linear.py tests/test_conv.py

test_block:
	uv run pytest -s tests/test_linear_block.py tests/test_conv_block.py

###
test_mlp:
	uv run pytest -s tests/test_mlp.py tests/test_resmlp.py

test_resnet:
	uv run pytest -s tests/test_resnet.py

test_autoencoder:
	uv run pytest -s tests/test_autoencoder.py

### 
test_datasets:
	uv run pytest -s tests/test_datasets.py

###
test_preprocessing:
	uv run pytest -s tests/test_preprocessing.py
