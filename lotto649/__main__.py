"""Command-line entry point for the Lotto649 selector."""
import sys

from .cli import main


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[1:] if args[:1] == ["select"] else args)
