"""Generate API reference pages for mkdocstrings."""

from pathlib import Path

import mkdocs_gen_files

PACKAGE = "ofdtd"

nav = mkdocs_gen_files.Nav()

root = Path(__file__).resolve().parent.parent
src = root / "src" / PACKAGE

# Optional landing page so /reference/ is not a 404.
with mkdocs_gen_files.open("reference/index.md", "w") as fd:
    fd.write(
        "# API Reference\n\n"
        "This section is generated automatically from the public Python API.\n"
    )

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(root / "src").with_suffix("")
    doc_path = path.relative_to(root / "src").with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    if not parts:
        continue

    nav[parts] = doc_path.as_posix()

    ident = ".".join(parts)

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"::: {ident}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())