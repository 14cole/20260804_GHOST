# Bundled Python 3.6 support

`ghost_backend/execution/_dataclasses.py` is the unmodified `dataclasses.py` from Eric V. Smith's
dataclasses 0.8 backport, distributed under Apache License 2.0. The complete
license is included as `execution/DATACLASSES_LICENSE.txt` beside the module.

- Project: https://github.com/ericvsmith/dataclasses
- Release: https://pypi.org/project/dataclasses/0.8/
- Source wheel: `dataclasses-0.8-py3-none-any.whl`
- Wheel SHA-256: `0201d89fa866f68c8ebd9d08ee6ff50c0b255f8ec63a71c16fda7af82bb887bf`

Only the filename is changed. `ghost_backend/execution/runtime.py` selects this private module
on Python 3.6 and the standard library on newer Python versions. It does not
shadow the standard library or require a separate dataclasses installation.
