"""Convenience entry point for experiment comparison."""

from echoquality.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["compare", *__import__("sys").argv[1:]]))
