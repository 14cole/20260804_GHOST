"""Build the optional table evaluator with a C99 compiler on Windows or Linux."""
import os
from pathlib import Path
import subprocess
import shutil


def build():
    root = Path(__file__).resolve().parent
    output = root / ('ghost_table.dll' if os.name == 'nt' else 'libghost_table.so')
    compiler = os.environ.get('CC', 'gcc')
    # No fast-math: preserve the validated double-precision interpolation.
    flags = ['-O3', '-std=c99', '-shared']
    flags += ['-static-libgcc'] if os.name == 'nt' else ['-fPIC']
    resolved = shutil.which(compiler)
    if resolved is None:
        raise RuntimeError('No C compiler is available; the Python evaluator remains usable.')
    env = dict(os.environ, PATH=str(Path(resolved).resolve().parent)+os.pathsep+os.environ.get('PATH',''))
    startup = None
    if os.name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
    subprocess.run([resolved, *flags, str(root/'table.c'), '-o', str(output)],
                   env=env, startupinfo=startup, check=True)
    return output


if __name__ == '__main__':
    print(build())
