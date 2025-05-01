.PHONY: test
test_all:
	poetry run pytest -s
###
test_components:
		poetry run pytest -s tests/test_linear.py tests/test_conv.py tests/test_linear_block.py tests/test_conv_block.py

test_models:
	poetry run pytest -s tests/test_mlp.py tests/test_resnet.py tests/test_resmlp.py tests/test_autoencoder.py

###
test_layer:
	poetry run pytest -s tests/test_linear.py tests/test_conv.py

test_block:
	poetry run pytest -s tests/test_linear_block.py tests/test_conv_block.py

###
test_mlp:
	poetry run pytest -s tests/test_mlp.py tests/test_resmlp.py

test_resnet:
	poetry run pytest -s tests/test_resnet.py

test_autoencoder:
	poetry run pytest -s tests/test_autoencoder.py

### 
test_datasets:
	poetry run pytest -s tests/test_datasets.py

###
test_preprocessing:
	poetry run pytest -s tests/test_preprocessing.py