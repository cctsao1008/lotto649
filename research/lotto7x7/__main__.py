"""Lotto649 selection entry point."""
import argparse
import sys


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["select"]:
        from selection.cli import main as select_main

        return select_main(argv[1:])
    if argv[:1] == ["research"]:
        from .legacy.cli import main as research_main

        return research_main(argv[1:] or ["--help"])
    if argv and argv[0] not in ("-h", "--help"):
        from .legacy.cli import main as research_main

        return research_main(argv)

    parser = argparse.ArgumentParser(prog="python -m research.lotto7x7", description="Lotto649 statistics-first selection")
    commands = parser.add_subparsers()
    commands.add_parser("select", help="Rank candidate combinations for a target date")
    commands.add_parser("research", help="Open the historical research CLI")
    parser.print_help()


def gallery(*args, **kwargs):
    """Preserve the historical gallery import."""
    from .legacy.cli import gallery as render

    return render(*args, **kwargs)


if __name__ == "__main__":
    main()
