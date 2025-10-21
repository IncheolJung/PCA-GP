import os
import multiprocessing
print(f"[{os.uname().nodename}] OMP_NUM_THREADS={os.getenv('OMP_NUM_THREADS')} nproc={multiprocessing.cpu_count()}")
