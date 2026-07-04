"""Generate the API reference pages and navigation.

Run by the mkdocs-gen-files plugin during ``properdocs build`` / ``properdocs serve``. It
is not meant to be run or imported directly. Only the public modules are
documented: private modules (``_project``, ``_rasterize``) and ``distillation``
are skipped.
"""

from pathlib import Path

try:
    import mkdocs_gen_files
except ImportError:
    pass  # not running in a docs environment, nothing to generate
else:
    # Modules kept out of the reference: private internals and the unstable
    # distillation module.
    SKIP_STEMS = {"distillation", "__main__", "__pycache__"}

    def skip(stem: str) -> bool:
        if stem in SKIP_STEMS:
            return True
        return stem.startswith("_") and stem != "__init__"

    nav_lines = ["* [Overview](index.md)"]

    for path in sorted(Path("splax").rglob("*.py")):
        parts = tuple(path.with_suffix("").parts)
        if any(skip(p) for p in parts):
            continue

        doc_path = path.with_suffix(".md")
        full_doc_path = Path("api", doc_path)

        if parts[-1] == "__init__":
            parts = parts[:-1]
            doc_path = doc_path.with_name("index.md")
            full_doc_path = full_doc_path.with_name("index.md")

        ident = ".".join(parts)
        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            fd.write(f"::: {ident}\n")
        mkdocs_gen_files.set_edit_path(full_doc_path, path)

        nav_lines.append(f"* [{ident}]({doc_path.as_posix()})")

    with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
        nav_file.write("\n".join(nav_lines) + "\n")
