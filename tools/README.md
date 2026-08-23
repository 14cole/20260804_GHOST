# Bundled tools

GRIM is the application host. Each computational tool remains a self-contained
subtree here so its standalone scripts, tests, examples, and documentation do
not become mixed into the GRIM plotting implementation.

- `GHOST/` is the complete 2-D, BoR, and feature-physics tool embedded in the
  GRIM **GHOST** tab.
- `FREDDY/` is the complete planar material-stack and impedance tool embedded
  in the GRIM **FREDDY** tab.

An embedded tab must call the tool's authoritative implementation from its own
subtree. Do not make a second copy of solver or feature physics inside
`GRIM_Revised_2`.

Both tools retain standalone launchers. FREDDY exports IBC and material CSV
files for GHOST; it does not produce `.grim` RCS datasets.
