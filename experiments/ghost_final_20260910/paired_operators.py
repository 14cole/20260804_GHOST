"""Finalize two polarized operators from each shared geometry tile query."""
from streamed_operator import *
import os,tempfile


class SpooledOperator(StreamedOperator):
    """Keep finalized compressed tiles on disk until this polarization is needed."""
    def __init__(self,*args,**kwargs):
        directory=Path(kwargs.pop('directory')).resolve()
        super().__init__(*args,**kwargs)
        fd,path=tempfile.mkstemp(prefix='ghost-tm-',suffix='.bin',dir=str(directory))
        self.path=Path(path).resolve()
        if self.path.parent!=directory:
            os.close(fd);raise RuntimeError('Unexpected spool location.')
        self.file=os.fdopen(fd,'w+b');self.records={};self.spool_bytes=0
        self.loaded=False
    def add_tile(self,i,j,raw,tail):
        super().add_tile(i,j,raw,tail)
        payload=self.tiles.pop((i,j));records=[]
        for value in payload:
            if value is None:records.append(None);continue
            records.append((value.shape,self.file.tell(),value.size))
            value.tofile(self.file);self.spool_bytes+=value.nbytes
        self.records[i,j]=records;self.tiles[i,j]=None
    def load(self):
        if self.loaded:return
        if self.file is None:raise ValueError('Compressed spool is closed.')
        self.file.flush()
        for key,records in self.records.items():
            self.checkpoint();payload=[]
            for record in records:
                if record is None:payload.append(None);continue
                shape,offset,count=record;self.file.seek(offset)
                value=np.fromfile(self.file,dtype=complex,count=count)
                if value.size!=count:raise IOError('Truncated compressed spool.')
                payload.append(value.reshape(shape))
            self.tiles[key]=tuple(payload)
        self.records.clear();self.loaded=True;self.close()
        self.evidence['spooled_bytes']=self.spool_bytes
    def _get(self,rows,cols):
        if not self.loaded:raise ValueError('Load the compressed spool before querying it.')
        return super()._get(rows,cols)
    def matmul(self,b,trans=0):
        if not self.loaded:raise ValueError('Load the compressed spool before multiplying it.')
        return super().matmul(b,trans)
    def close(self):
        if getattr(self,'file',None) is not None:self.file.close();self.file=None
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
        for i,rows in enumerate(operators[0].groups):
            for j,cols in enumerate(operators[0].groups):
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
