"""Generate API reference pages for mkdocstrings."""

from pathlib import Path

import mkdocs_gen_files

PACKAGE = "ofdtd"
SRC_ROOT = Path("src") / PACKAGE

nav = mkdocs_gen_files.Nav()

for path in sorted(SRC_ROOT.rglob("*.py")):
    if path.name == "__main__.py":
        continue

    module_path = path.relative_to("src").with_suffix("")
    doc_path = path.relative_to("src").with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        full_doc_path = full_doc_path.with_name("index.md")

    if not parts:
        continue

    nav[parts] = full_doc_path.as_posix()

    ident = ".".join(parts)

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"::: {ident}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
