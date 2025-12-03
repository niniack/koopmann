### Installation

This project uses `uv` for dependecy management. You can find `uv` installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer).

By default, `uv` has a default cache directory at `~/.cache/uv`. Once you have `uv` installed, you can verify this by running `uv cache dir`. 

If you want to use a custom cache directory (for example, on a different disk), set the `uv` relevant environment variables beforehand. You can set these variables in the `.env` file. There is an `.env.example` file, which you can copy into `.env`.

```bash
cp .env.example .env
```

Here, you can set directories and credentials (wandb, hugging-face). Of course, the `.env` file should not tracked by git (it is already in .gitignore!) Then, let `uv` know about the environment file.

```bash
export UV_ENV_FILE=$(pwd)/.env
```

Now, `uv` and code in this project will respect your environment variables.

```bash
uv sync # Installs required packages in a virtual env
```

### Running

If you chose to tell `uv` about your environment file, you can just use

```bash
uv run python script.py
```

Otherwise, you can use a flag

```bash
uv run --env-file .env python script.py
```
