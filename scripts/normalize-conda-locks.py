#!/usr/bin/env python3
"""Normalize conda-lock files written by pixi-to-conda-lock, in place.

Usage:
    scripts/normalize-conda-locks.py <pixi.lock> <conda-lock.yml>...

Two fixes:

1. Platform aliases. pixi.lock format v7 gives platforms with extra virtual
   packages (the named `workspace.platforms` entries, e.g. `linux-64-cuda12`;
   older pixi used `p1`, `p2`) their own name, with the real platform in `subdir`.
   pixi-to-conda-lock (<= 0.4.5) copies the alias into its output, so
   conda-lock consumers asking for `linux-64` find nothing. This rewrites
   `linux-64-cuda12` -> `linux-64` etc.

2. Ordering. pixi-to-conda-lock writes the `package:` list and the metadata
   platform lists in no fixed order, so every regeneration churns thousands of
   lines even when no pin changed. Package entries are sorted by (name,
   platform, manager, url) and platform lists alphabetically; each entry's text
   is kept byte-for-byte.
"""
import re
import sys

import yaml


def alias_map(pixi_lock_path):
    """Map platform alias -> subdir, from the top-level `platforms:` list of a v7 pixi.lock."""
    with open(pixi_lock_path) as f:
        data = yaml.safe_load(f)
    platforms = data.get("platforms")
    if not isinstance(platforms, list):
        return {}  # pre-v7 lock: no aliases
    return {p["name"]: p["subdir"] for p in platforms if "subdir" in p and p["subdir"] != p["name"]}


def fix_aliases(text, aliases):
    if not aliases:
        return text, 0
    names = "|".join(re.escape(a) for a in aliases)
    # The three places a platform name appears in conda-lock output:
    #   metadata.content_hash keys ("    p1: ..."), metadata.platforms items ("  - p1"),
    #   and each package's "  platform: p1".
    pattern = re.compile(rf"^(\s*platform: |\s*- |\s*)({names})(:\s|\s*$)", re.MULTILINE)
    return pattern.subn(lambda m: m.group(1) + aliases[m.group(2)] + m.group(3), text)


def entry_key(entry):
    def field(name):
        m = re.search(rf"^  {name}: (.*)$", entry, re.MULTILINE)
        return m.group(1) if m else ""
    return (field("name"), field("platform"), field("manager"), field("url"))


def sort_packages(text):
    """Sort the entries of the top-level `package:` list, which must be the last top-level key."""
    head, sep, body = text.partition("\npackage:\n")
    if not sep or re.search(r"^[A-Za-z]", body, re.MULTILINE):
        raise ValueError("unexpected layout: `package:` missing or not the last top-level key")
    entries = re.split(r"(?m)^(?=- )", body)
    if entries[0]:
        raise ValueError("unexpected text before the first package entry")
    entries = [e if e.endswith("\n") else e + "\n" for e in entries[1:]]
    return sort_metadata_platforms(head) + sep + "".join(sorted(entries, key=entry_key))


def sort_metadata_platforms(head):
    """Sort the per-platform lines under metadata.content_hash and metadata.platforms."""
    def sort_block(m):
        lines = m.group(2).splitlines(keepends=True)
        return m.group(1) + "".join(sorted(lines))
    head = re.sub(r"(?m)(^  content_hash:\n)((?:^    \S.*\n)+)", sort_block, head + "\n")
    head = re.sub(r"(?m)(^  platforms:\n)((?:^  - .*\n)+)", sort_block, head)
    return head[:-1]


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    aliases = alias_map(sys.argv[1])
    if aliases:
        print("aliases:", ", ".join(f"{a} -> {s}" for a, s in sorted(aliases.items())))
    for path in sys.argv[2:]:
        with open(path) as f:
            text = f.read()
        new_text, n = fix_aliases(text, aliases)
        new_text = sort_packages(new_text)
        if new_text != text:
            with open(path, "w") as f:
                f.write(new_text)
        print(f"{path}: {n} alias replacements, packages sorted")


if __name__ == "__main__":
    main()
