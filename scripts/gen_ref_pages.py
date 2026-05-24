"""Generate nested API reference pages for mkdocstrings."""

from pathlib import Path

import mkdocs_gen_files

PACKAGE = "ofdtd"

root = Path(__file__).resolve().parent.parent
src_root = root / "src"
package_root = src_root / PACKAGE

nav = mkdocs_gen_files.Nav()

for path in sorted(package_root.rglob("*.py")):
    if path.name == "__main__.py":
        continue

    module_path = path.relative_to(src_root).with_suffix("")
    parts = list(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = Path(*parts) / "index.md"
    else:
        doc_path = Path(*parts).with_suffix(".md")

    if not parts:
        continue

    full_doc_path = Path("reference") / doc_path
    ident = ".".join(parts)

    # This creates the nested folder/file structure in the sidebar.
    nav[tuple(parts)] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"# `{ident}`\n\n")
        fd.write(f"::: {ident}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())