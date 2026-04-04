from typing import Sequence

from .settings.env import Env
from loguru import logger


# The CLI package owns the command surface so the library package can stay
# import-focused under the new two-package src layout.
def main(argv: Sequence[str] | None = None) -> int:
    env = Env()  # ty:ignore[missing-argument]
    logger.info(f"using llm endpoint: {env.openai_compatible_base_url}")
    logger.info(f"using llm model: {env.llm_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
