#!/usr/bin/env python3
"""
Every Python source file must be pure ASCII.

A UTF-8 source file that passes through a Windows editor, an FTP client in
text mode, or any tool that re-encodes on save comes out the other side in a
local codepage. The em dash this repo used in comments and error messages
becomes a lone 0x97 byte, which Python 3 rejects outright:

    SyntaxError: (unicode error) 'utf-8' codec can't decode byte 0x97
    in position 0: invalid start byte

The file is then unimportable on the machine it was copied to, and the error
points at a decorative character rather than at anything meaningful. Nothing
here needs non-ASCII -- the characters were box-drawing rules in comment
banners, em dashes in prose, and Greek letters in docstrings -- so the cheapest
fix is to not have any.

Data files are exempt: .geo and .csv inputs are read with an explicit encoding
and are the user's, not ours.

Usage:
    python tests/test_source_is_ascii.py
"""

import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    offenders = []
    scanned = 0
    for path in sorted(REPO.rglob("*.py")):
        if any(part in {".git", "__pycache__"} for part in path.parts):
            continue
        scanned += 1
        raw = path.read_bytes()
        try:
            raw.decode("ascii")
        except UnicodeDecodeError:
            text = raw.decode("utf-8", errors="replace")
            for number, line in enumerate(text.splitlines(), start=1):
                for column, char in enumerate(line):
                    if ord(char) > 127:
                        offenders.append((
                            path.relative_to(REPO), number, column,
                            char, unicodedata.name(char, "?"),
                        ))

    print(f"scanned {scanned} Python file(s)")
    if not offenders:
        print("all pure ASCII")
        return 0

    print(f"\n{len(offenders)} non-ASCII character(s):\n")
    for rel, number, column, char, name in offenders[:40]:
        try:
            cp1252 = f"0x{char.encode('cp1252').hex()}"
        except UnicodeEncodeError:
            cp1252 = "not representable"
        print(f"  {rel}:{number}:{column}  {char!r}  U+{ord(char):04X}  "
              f"{name}  (cp1252 {cp1252})")
    if len(offenders) > 40:
        print(f"  ... and {len(offenders) - 40} more")
    print("\nReplace them with ASCII equivalents; see the module docstring.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
