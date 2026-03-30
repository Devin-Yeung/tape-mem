# Quality Assurance

Run following check before delivering the code:

- [ ] linter via `uv run ruff check`
- [ ] type check via `uv run ty check`

As for type errors, try to fix the root of cause instead of suppressing it.
If you are very confident that the type error is a false positive,
you can ignore it with a comment like `# type: ignore[error-code]` to specify the error code you want to ignore.
