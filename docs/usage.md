# CLI Usage

The `tape-mem` CLI runs the full-context memory agent demo on an EventQA example. Make sure you have set up the required
python environment in the [development](development/setup-python.md) section before running the CLI.

## Environment Variables

The CLI requires three environment variables:

| Variable                     | Description                                                 | Required |
| ---------------------------- | ----------------------------------------------------------- | -------- |
| `OPENAI_COMPATIBLE_BASE_URL` | Base URL of your LLM API (e.g. `https://api.openai.com/v1`) | Yes      |
| `OPENAI_COMPATIBLE_API_KEY`  | API key for the LLM service                                 | Yes      |
| `LLM_MODEL`                  | Model name to use (e.g. `gpt-4o-mini`)                      | Yes      |

> **Note:** `HF_ENDPOINT=https://hf-mirror.com` will be used if `HF_ENDPOINT` is not set for faster Hugging Face
> downloads, so no extra setup is needed.
> Or, to override the Hugging Face mirror with:
>
> ```bash
> HF_ENDPOINT=<endpoint>
> ```

---

## Run

Make sure to set the required environment variables, then you can run the CLI with:

```bash
# if you prefer using uv, you can run with:
uv run tape-mem

# or, if you have tape-mem installed in your Python environment, simply run:
tape-mem
```
