"""Shared GRIM application colors, independent of the main GUI and solvers."""

from __future__ import annotations

APPLICATION_PALETTES: dict[str, dict[str, object]] = {
    "Colorful": {
        "is_dark": True,
        "win_bg": "#111827",
        "panel_bg": "#19162f",
        "text": "#f8fafc",
        "head_bg": "#312e81",
        "border": "#6d28d9",
        "hover": "#0e7490",
        "checked_bg": "#7c3aed",
        "checked_border": "#22d3ee",
        "grid": "#475569",
        "muted": "#c4b5fd",
        "fg": "#f8fafc",
        "plot_line_freq": "#22d3ee",
        "plot_line_angle": "#f472b6",
        "plot_worst": "#fbbf24",
        "layer_colors": (
            "#7c3aed", "#0891b2", "#db2777", "#d97706",
            "#4f46e5", "#0e7490", "#9333ea", "#0284c7",
        ),
    },
    "Light": {
        "is_dark": False,
        "win_bg": "#f1f5f9",
        "panel_bg": "#ffffff",
        "text": "#0f172a",
        "head_bg": "#dbeafe",
        "border": "#94a3b8",
        "hover": "#bfdbfe",
        "checked_bg": "#2563eb",
        "checked_border": "#1d4ed8",
        "grid": "#cbd5e1",
        "muted": "#475569",
        "fg": "#0f172a",
        "plot_line_freq": "#0369a1",
        "plot_line_angle": "#6d28d9",
        "plot_worst": "#b45309",
        "layer_colors": (
            "#1d4ed8", "#0369a1", "#4f46e5", "#0e7490",
            "#2563eb", "#475569", "#7c3aed", "#0284c7",
        ),
    },
    "Dark": {
        "is_dark": True,
        "win_bg": "#0f172a",
        "panel_bg": "#0b1222",
        "text": "#dbeafe",
        "head_bg": "#172554",
        "border": "#1e3a8a",
        "hover": "#1d4ed8",
        "checked_bg": "#2563eb",
        "checked_border": "#3b82f6",
        "grid": "#475569",
        "muted": "#94a3b8",
        "fg": "#dbeafe",
        "plot_line_freq": "#38bdf8",
        "plot_line_angle": "#a78bfa",
        "plot_worst": "#fbbf24",
        "layer_colors": (
            "#1e3a8a", "#1d4ed8", "#172554", "#2563eb",
            "#1e40af", "#3b82f6", "#334155", "#0284c7",
        ),
    },
    "Neutral Dark": {
        "is_dark": True,
        "win_bg": "#111827",
        "panel_bg": "#1f2937",
        "text": "#f3f4f6",
        "head_bg": "#273244",
        "border": "#374151",
        "hover": "#475569",
        "checked_bg": "#2563eb",
        "checked_border": "#60a5fa",
        "grid": "#374151",
        "muted": "#9ca3af",
        "fg": "#f3f4f6",
        "success": "#34d399",
        "warning": "#fbbf24",
        "danger": "#f87171",
        "plot_line_freq": "#38bdf8",
        "plot_line_angle": "#c4b5fd",
        "plot_worst": "#fbbf24",
        "layer_colors": (
            "#2563eb", "#0e7490", "#7c3aed", "#b45309",
            "#047857", "#be185d", "#4338ca", "#475569",
        ),
    },
    "Raytheon": {
        "is_dark": False,
        "win_bg": "#d9d9d6",
        "panel_bg": "#ffffff",
        "text": "#000000",
        "head_bg": "#d9d9d6",
        "border": "#63666a",
        "hover": "#63666a",
        "checked_bg": "#ce1126",
        "checked_border": "#ce1126",
        "grid": "#b1b3b3",
        "muted": "#63666a",
        "fg": "#000000",
        # Embedded application plots use primary/secondary colors. The brand's
        # tertiary colors are reserved for PowerPoint charts when needed.
        "plot_line_freq": "#000000",
        "plot_line_angle": "#ce1126",
        "plot_worst": "#63666a",
        "layer_colors": (
            "#000000", "#ce1126", "#63666a", "#b1b3b3",
            "#d9d9d6", "#000000", "#ce1126", "#63666a",
        ),
    },
}
DEFAULT_APPLICATION_PALETTE = "Dark"
APPLICATION_PALETTE_DESCRIPTIONS = {
    "Colorful": "Purple, cyan, and magenta dark application chrome",
    "Light": "Bright neutral application chrome with blue accents",
    "Dark": "GRIM blue/slate dark application chrome",
    "Neutral Dark": "Neutral slate surfaces, subtle borders, and blue actions",
    "Raytheon": "Official white, black, cool gray, and Red 186 chrome",
}
APPLICATION_PALETTE_SETTINGS_KEY = "appearance/application_palette"
LEGACY_APPLICATION_PALETTE_NAMES = {
    "Raytheon-inspired": "Raytheon",
}
# Official tertiary colors are intentionally not part of application chrome
# or embedded-tool plots. They are reserved for optional PowerPoint charts
# with enough series to require additional differentiation.
RAYTHEON_TERTIARY_PPT_CHART_COLORS = (
    "#7ba7bc",
    "#b7a99a",
    "#908cc2",
    "#9abeaa",
    "#efb661",
)
# Compatibility export for extensions/tests that used GRIM's former one fixed
# palette. It remains the exact default Dark palette.
BLUE_PALETTE = APPLICATION_PALETTES[DEFAULT_APPLICATION_PALETTE]


def normalize_application_palette_name(value: object) -> str:
    """Return a current palette name, including legacy-setting migration."""

    normalized = str(value).strip()
    normalized = LEGACY_APPLICATION_PALETTE_NAMES.get(
        normalized,
        normalized,
    )
    if normalized not in APPLICATION_PALETTES:
        return DEFAULT_APPLICATION_PALETTE
    return normalized
