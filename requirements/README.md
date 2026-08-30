# Reproducible Windows dependency bundle

The supported internal release target is 64-bit CPython 3.12 on Windows. The
exact reviewed dependency graph is in `constraints-windows-py312.txt`; do not
upgrade one package in isolation. A source release never contacts a package
index and the release gate refuses a build when its required installed stack
does not match that lock.

Prepare the reusable team wheelhouse once on a connected Windows CPython 3.12
machine:

```powershell
py -3.12 -m pip download --only-binary=:all: --dest wheelhouse `
  -r requirements\windows-py312.txt
py -3.12 requirements\wheelhouse_manifest.py generate wheelhouse `
  --constraints requirements\constraints-windows-py312.txt
py -3.12 requirements\wheelhouse_manifest.py verify wheelhouse `
  --constraints requirements\constraints-windows-py312.txt
```

Keep that wheelhouse alongside the release ZIP in the team distribution
location. The manifest utility is standard-library-only, requires exactly one
valid wheel at every locked version, binds the filename, `.dist-info`, package
metadata, and WHEEL compatibility tags to the same package identity, rejects
additional unlocked packages, and rejects wheels outside pip's 64-bit Windows
CPython 3.12 tag set before SHA-256 verifying every byte. It never contacts a
package index. Archive the wheelhouse only after `verify` passes.

On an offline destination, after extracting GRIM, verify the copied wheelhouse
again before installing:

```powershell
py -3.12 requirements\wheelhouse_manifest.py verify ..\wheelhouse `
  --constraints requirements\constraints-windows-py312.txt
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --no-index --find-links ..\wheelhouse `
  -r requirements\windows-py312.txt
.venv\Scripts\python.exe -m pip install --no-index --find-links ..\wheelhouse `
  --no-build-isolation -c requirements\constraints-windows-py312.txt -e .
.venv\Scripts\python.exe -m grim_diagnostics
```

Archive and checksum the wheelhouse whenever the lock changes. Because the
wheelhouse is an independently reusable dependency bundle, it is not placed in
or discovered from a developer checkout by the source-release builder.
