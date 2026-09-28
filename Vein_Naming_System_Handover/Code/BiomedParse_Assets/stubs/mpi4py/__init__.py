class _Comm:
    def Get_rank(self): return 0
    def Get_size(self): return 1
    def gather(self, data, root=0): return [data] if self.Get_rank() == root else None
    def bcast(self, data, root=0): return data
    def barrier(self): pass
    def Barrier(self): pass
    def Allreduce(self, sendbuf, recvbuf, op=None): pass
    def allreduce(self, data, op=None): return data

class _MPI:
    COMM_WORLD = _Comm()
    SUM = 'SUM'
    MAX = 'MAX'
    MIN = 'MIN'

MPI = _MPI()
