"""Standalone command-line interface for flake8-elegant-objects."""

import argparse
import ast
import sys

from .base import ElegantObjectsCore


def main() -> None:
    """Standalone command-line interface."""
    arguments = argparse.ArgumentParser(
        description="Check Python files for Elegant Objects violations"
    )
    arguments.add_argument("files", nargs="+", help="Python files to check")
    arguments.add_argument(
        "--show-source",
        action="store_true",
        help="Show source code context for violations",
    )

    args = arguments.parse_args()

    total_errors = 0

    for file_path in args.files:
        if not file_path.endswith(".py"):
            continue

        try:
            with open(file_path, encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source, filename=file_path)
            core = ElegantObjectsCore(tree)

            file_errors = 0
            violations = core.check_violations()

            for violation in violations:
                print(
                    f"{file_path}:{violation.line}:{violation.column}: {violation.message}"
                )

                if args.show_source:
                    lines = source.split("\n")
                    if 0 <= violation.line - 1 < len(lines):
                        print(f"    {lines[violation.line - 1].strip()}")
                    print()

                file_errors += 1
                total_errors += 1

            if file_errors == 0:
                print(f"{file_path}: No violations found ✓")

        except Exception as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)

    if total_errors > 0:
        print(f"\nTotal violations found: {total_errors}")
        sys.exit(1)
    else:
        print("\nAll files comply with Elegant Objects principles! ✓")


if __name__ == "__main__":
    main()
