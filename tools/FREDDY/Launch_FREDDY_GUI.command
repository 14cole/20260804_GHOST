#!/bin/zsh

set -u

launcher_dir="${0:A:h}"
cd "$launcher_dir" || exit 1
check_only=0
if [[ "${1:-}" == "--check" ]]; then
    check_only=1
fi

pause_before_exit() {
    if (( ! check_only )); then
        read "?Press Return to close."
    fi
}

if [[ ! -f "impedance_gui.py" ]]; then
    echo "ERROR: impedance_gui.py was not found."
    echo "Keep the complete tools/FREDDY folder together."
    pause_before_exit
    exit 1
fi

matplotlib_dir="${TMPDIR:-/tmp}/freddy-matplotlib"
mkdir -p "$matplotlib_dir"
export MPLCONFIGDIR="$matplotlib_dir"

typeset -a python_candidates
python_candidates=(
    "$launcher_dir/.venv/bin/python3"
)
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    python_candidates+=("$VIRTUAL_ENV/bin/python3")
fi
python_candidates+=(
    /Library/Frameworks/Python.framework/Versions/Current/bin/python3
    /Library/Frameworks/Python.framework/Versions/*/bin/python3(N)
    /opt/homebrew/bin/python3
    /usr/local/bin/python3
)
path_python="$(command -v python3 2>/dev/null || true)"
if [[ -n "$path_python" ]]; then
    python_candidates+=("$path_python")
fi

diagnostic_file="${TMPDIR:-/tmp}/freddy-gui-launch-$$.log"
: > "$diagnostic_file"
selected_python=""
typeset -A tried_python
for candidate in "${python_candidates[@]}"; do
    [[ -x "$candidate" ]] || continue
    resolved_candidate="${candidate:A}"
    [[ -n "${tried_python[$resolved_candidate]:-}" ]] && continue
    tried_python[$resolved_candidate]=1
    echo "--- $candidate ---" >> "$diagnostic_file"
    if "$candidate" -c "import numpy, scipy; from ibc.ui import QT_AVAILABLE, MPL_AVAILABLE; assert QT_AVAILABLE, 'PySide6 is not available'; assert MPL_AVAILABLE, 'The Matplotlib Qt backend is not available'" >> "$diagnostic_file" 2>&1; then
        selected_python="$candidate"
        break
    fi
done

if [[ -z "$selected_python" ]]; then
    echo "ERROR: No installed Python interpreter could import the FREDDY GUI."
    echo
    echo "Interpreter diagnostics:"
    cat "$diagnostic_file"
    echo
    echo "From this folder, install FREDDY's dependencies with:"
    echo "    /path/to/python3 -m pip install -r requirements.txt"
    echo
    pause_before_exit
    exit 1
fi

if (( check_only )); then
    echo "FREDDY GUI will use: $selected_python"
    exit 0
fi

exec "$selected_python" "impedance_gui.py"
