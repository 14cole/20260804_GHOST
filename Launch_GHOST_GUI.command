#!/bin/zsh

set -u

launcher_dir="${0:A:h}"
cd "$launcher_dir" || exit 1

if [[ ! -f "Backend/ghost_gui.py" ]]; then
    echo "ERROR: Backend/ghost_gui.py was not found."
    read "?Press Return to close."
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: Python 3 was not found."
    echo "Install Python 3.10 or newer, then run this launcher again."
    read "?Press Return to close."
    exit 1
fi

if ! python3 "Backend/ghost_gui.py" --check >/dev/null 2>&1; then
    echo "ERROR: One or more GHOST GUI dependencies could not be imported."
    echo
    echo "Install them with:"
    echo "    python3 -m pip install numpy scipy matplotlib PySide6"
    echo
    read "?Press Return to close."
    exit 1
fi

exec python3 "Backend/ghost_gui.py"
