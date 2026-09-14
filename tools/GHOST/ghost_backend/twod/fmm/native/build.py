"""Build the vendored FMM2D sources with a complete GNU Fortran installation.

Run: python -m ghost_backend.twod.fmm.native.build
Set FC to gfortran's full path when it is not on PATH. No network downloads,
installation, architecture-specific CPU flags, or import-time builds occur.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os,platform,subprocess,tempfile,json,hashlib,shutil


def main():
    root=Path(__file__).resolve().parent
    fc=os.environ.get('FC') or shutil.which('gfortran')
    if not fc:raise RuntimeError('Install a complete GNU Fortran toolchain and put gfortran on PATH or set FC.')
    env=dict(os.environ)
    env['PATH']=str(Path(fc).resolve().parent)+os.pathsep+env.get('PATH','')
    flags=['-O3','-funroll-loops','-std=legacy','-fopenmp','-w']
    system=platform.system()
    if system=='Windows':
        name='ghost_fmm.windows-amd64.dll'
        if platform.machine().lower() not in ('amd64','x86_64'):raise RuntimeError('The Windows build requires x86-64.')
        link=['-shared','-static','-static-libgfortran','-static-libgcc','-Wl,--no-insert-timestamp']
    elif system=='Darwin':
        name='libghost_fmm.dylib';flags+=['-fPIC'];link=['-dynamiclib']
    else:name='libghost_fmm.so';flags+=['-fPIC'];link=['-shared']
    sources=sorted((root/'vendor').rglob('*.f')) + sorted(root.glob('*.f90'))
    with tempfile.TemporaryDirectory(prefix='ghost-fmm-build-') as directory:
        def compile(source):
            out=Path(directory)/(source.stem+'.o')
            subprocess.run([fc,*flags,'-J'+directory,'-c',str(source),'-o',str(out)],env=env,check=True)
            return out
        with ThreadPoolExecutor(max_workers=min(8,os.cpu_count() or 1)) as pool:
            objects=list(pool.map(compile,sources))
        output=Path(directory)/name
        subprocess.run([fc,*flags,*link,*map(str,objects),'-o',str(output)],env=env,check=True)
        # Check exported symbols in a fresh process before replacing the library.
        subprocess.run([__import__('sys').executable,'-c',
            'import ctypes,sys; d=ctypes.CDLL(sys.argv[1]); d.hfmm2d_; d.ghost_fmm_create; d.omp_set_num_threads_',str(output)],check=True,env=env)
        shutil.copy2(output,root/name)
    info=dict(upstream=json.loads((root/'vendor/UPSTREAM.json').read_text()),
        compiler=subprocess.check_output([fc,'--version'],env=env,text=True).splitlines()[0],
        flags=flags,link_flags=link,sha256=hashlib.sha256((root/name).read_bytes()).hexdigest())
    (root/'build-info.json').write_text(json.dumps(info,indent=2))
    print(root/name)

if __name__=='__main__':main()
