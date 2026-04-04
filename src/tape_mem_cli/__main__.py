from typing import Sequence


# The CLI package owns the command surface so the library package can stay
# import-focused under the new two-package src layout.
def main(argv: Sequence[str] | None = None) -> int:
    # TODO: Implement command parsing once the public CLI contract is defined.
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
