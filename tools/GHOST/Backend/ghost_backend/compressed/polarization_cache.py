"""Finalize two polarized operators from each shared geometry tile query."""
from ghost_backend.compressed.operator import StreamedOperator
from pathlib import Path
import numpy as np
import os,tempfile,hashlib


class SpooledOperator(StreamedOperator):
    """Keep finalized compressed tiles on disk until this polarization is needed."""
    def __init__(self,*args,**kwargs):
        directory=Path(kwargs.pop('directory')).resolve()
        assemble=kwargs.pop('assemble',True)
        super().__init__(*args,assemble=False,**kwargs)
        fd,path=tempfile.mkstemp(prefix='ghost-tm-',suffix='.bin',dir=str(directory))
        self.path=Path(path).resolve()
        if self.path.parent!=directory:
            os.close(fd);raise RuntimeError('Unexpected spool location.')
        self.file=os.fdopen(fd,'w+b');self.records={};self.spool_bytes=0
        self.loaded=False
        if assemble:
            try:self.assemble_tiles(args[0])
            except BaseException:
                self.close();raise
    def add_tile(self,i,j,raw,tail):
        super().add_tile(i,j,raw,tail)
        payload=self.tiles.pop((i,j));records=[]
        for value in payload:
            if value is None:records.append(None);continue
            raw=value.tobytes(order='C')
            records.append((value.shape,self.file.tell(),value.size,hashlib.sha256(raw).digest()))
            self.file.write(raw);self.spool_bytes+=value.nbytes
        self.records[i,j]=records;self.tiles[i,j]=None
    def load(self):
        if self.loaded:return
        if self.file is None:raise ValueError('Compressed spool is closed.')
        try:
            self.file.flush()
            for key,records in self.records.items():
                self.checkpoint();payload=[]
                for record in records:
                    if record is None:payload.append(None);continue
                    shape,offset,count,digest=record;self.file.seek(offset)
                    value=np.fromfile(self.file,dtype=complex,count=count)
                    if value.size!=count:raise IOError('Truncated compressed spool.')
                    if hashlib.sha256(value.tobytes()).digest()!=digest:raise IOError('Compressed spool checksum mismatch.')
                    payload.append(value.reshape(shape))
                self.tiles[key]=tuple(payload)
        except BaseException:


            self.tiles.clear();self.records.clear();self.close()
            raise
        self.records.clear();self.loaded=True;self.close()
        self.evidence['spooled_bytes']=self.spool_bytes
    def _get(self,rows,cols):
        if not self.loaded:raise ValueError('Load the compressed spool before querying it.')
        return super()._get(rows,cols)
    def matmul(self,b,trans=0):
        if not self.loaded:raise ValueError('Load the compressed spool before multiplying it.')
        return super().matmul(b,trans)
    def iter_tiles(self):
        if not self.loaded:raise ValueError('Load the compressed spool before reading its tiles.')
        return super().iter_tiles()
    def close(self):
        file,self.file=getattr(self,'file',None),None
        try:
            if file is not None:file.close()
        finally:
            if getattr(self,'path',None) is not None and self.path.exists():self.path.unlink()
    def __del__(self):
        try:self.close()
        except OSError:pass


def build_pair(oracle,coordinates,tile=512,budget=512*1024**2,checkpoint=None,spool_directory=None):
    operators=[]
    try:
        for index,o in enumerate(oracle.oracles):
            cls=SpooledOperator if index==1 and spool_directory is not None else StreamedOperator
            extra={'directory':spool_directory} if cls is SpooledOperator else {}
            operators.append(cls(o,coordinates,tile=tile,budget=budget,checkpoint=checkpoint,assemble=False,**extra))
        for j,cols in enumerate(operators[0].groups):
            if hasattr(oracle,'prepare_columns'):oracle.prepare_columns(cols)
            for i,rows in enumerate(operators[0].groups):
                values=oracle.get_with_error(rows,cols)
                for index in range(2):
                    target=operators[index]
                    target.budget=budget-operators[1-index].bytes
                    target.add_tile(i,j,*values[index])
                    values[index]=None
        for op,source in zip(operators,oracle.oracles):op.finalize(source)
    except BaseException:
        for op in operators:
            if isinstance(op,SpooledOperator):op.close()
        raise
    return operators
