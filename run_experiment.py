"""Convenience entry point for Windows/macOS.

Equivalent to: python -m echoquality run ...
"""

from echoquality.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["run", *__import__("sys").argv[1:]]))
