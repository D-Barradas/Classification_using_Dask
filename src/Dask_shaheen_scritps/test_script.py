import numpy as np,time
from dask.distributed import Client
import dask.array as da
import os
jobid=os.getenv('SLURM_JOBID')
client= Client(scheduler_file='scheduler_%s.json' %jobid)

print(client)

x = da.random.normal(10, 0.1, size=(20000, 20000),   # 400 million element array 
                                      chunks=(1000, 1000))   # Cut into 1000x1000 sized chunks
y = x.mean(axis=0)[::100]                            # Perform NumPy-style operations

print(x.nbytes/1e9,"GB")
start=time.time()
print(y.compute())
end=time.time()

print('elapsed=',end-start)

client.close()
