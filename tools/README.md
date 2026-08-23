# Bundled tools

GRIM is the application host. Each computational tool remains a self-contained
subtree here so its standalone scripts, tests, examples, and documentation do
not become mixed into the GRIM plotting implementation.

- `GHOST/` is the complete active 2-D, BoR, and feature-physics tool.
- `FREDDY/` is the tracked location for the planned future integration.

An embedded tab must call the tool's authoritative implementation from its own
subtree. Do not make a second copy of solver or feature physics inside
`GRIM_Revised_2`.
