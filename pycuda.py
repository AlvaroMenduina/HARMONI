"""

                  ----- PyCUDA Experiment -----

Trying to generate training examples on the GPU

"""

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np

if __name__ == "__main__":

    a = np.random.randn(4, 4).astype(np.float32)
    a_gpu = cuda.mem_alloc(a.nbytes)
    cuda.memcpy_htod(a_gpu, a)

    mod = SourceModule("""
        __global__ void doublify(float *a){
        int idx = threadIdx.x + threadIdx.y*4;
        a[idx] *= 2;}
        """)
