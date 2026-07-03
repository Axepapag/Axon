"""Entry point: python -m runtime.table <command>."""
import sys

from runtime.table.cli import main

if __name__ == "__main__":
    sys.exit(main())
