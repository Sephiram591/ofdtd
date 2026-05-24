"""Generate nested API reference pages for mkdocstrings.

This creates one docs page for every package folder and every Python file under
src/ofdtd/.

Example output:

reference/
└── ofdtd/
    ├── index.md      # src/ofdtd/__init__.py
    ├── config.md     # src/ofdtd/config.py
    ├── core.md       # src/ofdtd/core.py
    └── subpackage/
        ├── index.md  # src/ofdtd/subpackage/__init__.py
        └── module.md # src/ofdtd/subpackage/module.py
"""

from pathlib import Path

import mkdocs_gen_files

PACKAGE = "ofdtd"

root = Path(__file__).resolve().parent.parent
src_root = root / "src"

nav = mkdocs_gen_files.Nav()


def format_title(name: str) -> str:
    """Convert a module or package name into a readable nav title."""
    if name == PACKAGE:
        return PACKAGE
    return name.replace("_", " ").title()


for path in sorted((src_root / PACKAGE).rglob("*.py")):
    if path.name == "__main__.py":
        continue

    module_path = path.relative_to(src_root).with_suffix("")
    parts = list(module_path.parts)

    if parts[-1] == "__init__":
        # Folder/package page.
        parts = parts[:-1]
        doc_path = Path(*parts) / "index.md"
    else:
        # Normal Python module page.
        doc_path = Path(*parts).with_suffix(".md")

    full_doc_path = Path("reference") / doc_path
    ident = ".".join(parts)

    # Make the visible nav titles nicer while preserving the import path.
    nav_parts = tuple(format_title(part) for part in parts)

    nav[nav_parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"# `{ident}`\n\n")
        fd.write(f"::: {ident}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))


with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())