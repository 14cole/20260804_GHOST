# FREDDY integration slot

This directory is reserved for the future FREDDY tool integration. FREDDY
should remain one self-contained tool subtree, parallel to `tools/GHOST`, while
GRIM remains the host application.

The intended integration boundary is:

- expose a reusable FREDDY `QWidget` workspace for a GRIM tab;
- keep FREDDY's computational implementation authoritative inside this
  subtree rather than copying it into GRIM;
- publish generated `.grim` artifacts through the host's existing dataset
  loader;
- expose a running-job/close guard so GRIM cannot close during active work;
- keep a thin standalone FREDDY launcher if solver-only use is still useful.

No FREDDY solver or GUI has been added by this branch. This tracked location
only establishes the one-checkout organization for that later integration.
