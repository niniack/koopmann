#### Repository Layout

Repository layout. 

### Installation

This project uses `uv` for dependecy management. You can find `uv` installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer).

Out of the box, `uv` has a default cache directory at `~/.cache/uv`. Once you have `uv` installed, you can verify this by running `uv cache dir`. 

If you want to use a custom cache directory (for example, on a different disk), set the `uv`-relevant environment variables beforehand. You can set these variables in an `.env` file which lives in the project root. This repository provides an `.env.example` file, which you can copy into `.env` and customize.

```bash
cp .env.example .env
```

In the `.env` file, you can set directories and credentials (wandb, hugging-face). The `.env` file should never be tracked by git (it is already in .gitignore!)

### Running

Pydantic handles environment variables