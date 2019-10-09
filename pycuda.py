"""

                  ----- PyCUDA Experiment -----

Trying to generate training examples on the GPU


http://www.orangeowlsolutions.com/archives/526
"""

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time

# PARAMETERS
Z = 1.25                    # Strength of the aberrations -> relates to the Strehl ratio
pix = 30                    # Pixels to crop the PSF
N_PIX = 256                 # Pixels for the Fourier arrays
RHO_APER = 0.5              # Size of the aperture relative to the physical size of the Fourier arrays
RHO_OBSC = 0.15             # Central obscuration

def invert_mask(x, mask):
    """
    Takes a vector X which is the result of masking a 2D with the Mask
    and reconstructs the 2D array
    Useful when you need to evaluate a Zernike Surface and most of the array is Masked
    """
    N = mask.shape[0]
    ij = np.argwhere(mask==True)
    i, j = ij[:,0], ij[:,1]
    result = np.zeros((N, N))
    result[i,j] = x
    return result

def invert_mask_datacube(x, mask):
    """
    Takes a vector X which is the result of masking a 2D with the Mask
    and reconstructs the 2D array
    Useful when you need to evaluate a Zernike Surface and most of the array is Masked
    """
    M = x.shape[-1]
    N = mask.shape[0]
    ij = np.argwhere(mask==True)
    i, j = ij[:,0], ij[:,1]
    result = np.zeros((M, N, N)).astype(np.float32)
    for k in range(M):
        result[k,i,j] = x[:,k]
    return result

# ==================================================================================================================== #
#                                   Deformable Mirror - ACTUATOR MODEL functions
# ==================================================================================================================== #

def actuator_centres(N_actuators, rho_aper=RHO_APER, rho_obsc=RHO_OBSC, radial=True):
    """
    Computes the (Xc, Yc) coordinates of actuator centres
    inside a circle of rho_aper, assuming there are N_actuators
    along the [-1, 1] line

    :param N_actuators:
    :param rho_aper:
    :return:
    """

    x0 = np.linspace(-1., 1., N_actuators, endpoint=True)
    delta = x0[1] - x0[0]
    N_in_D = 2*RHO_APER/delta
    print('%.2f actuators in D' %N_in_D)
    max_freq = N_in_D / 2                   # Max spatial frequency we can sense
    xx, yy = np.meshgrid(x0, x0)
    x_f = xx.flatten()
    y_f = yy.flatten()

    act = []
    for x_c, y_c in zip(x_f, y_f):
        r = np.sqrt(x_c ** 2 + y_c ** 2)
        if r < rho_aper - delta/2 and r > rho_obsc + delta/2:
            act.append([x_c, y_c])

    if radial:
        for r in [rho_aper, rho_obsc]:
            N_radial = int(np.floor(2*np.pi*r/delta))
            d_theta = 2*np.pi / N_radial
            theta = np.linspace(0, 2*np.pi - d_theta, N_radial)
            # Super important to do 2Pi - d_theta to avoid placing 2 actuators in the same spot... Degeneracy
            for t in theta:
                act.append([r*np.cos(t), r*np.sin(t)])


    total_act = len(act)
    print('Total Actuators: ', total_act)
    return [act, delta], max_freq

def plot_actuators(centers):
    N_act = len(centers[0])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    circ1 = Circle((0,0), RHO_APER, linestyle='--', fill=None)
    circ2 = Circle((0,0), RHO_OBSC, linestyle='--', fill=None)
    ax.add_patch(circ1)
    ax.add_patch(circ2)
    for c in centers[0]:
        ax.scatter(c[0], c[1], color='red')
    ax.set_aspect('equal')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.title('%d actuators' %N_act)

def rbf_matrix(centres, rho_aper=RHO_APER, rho_obsc=RHO_OBSC):

    cent, delta = centres
    N_act = len(cent)
    matrix = np.empty((N_PIX, N_PIX, N_act))
    x0 = np.linspace(-1., 1., N_PIX, endpoint=True)
    xx, yy = np.meshgrid(x0, x0)
    rho = np.sqrt(xx ** 2 + yy ** 2)
    pupil = (rho <= rho_aper) & (rho >= rho_obsc)

    for k in range(N_act):
        xc, yc = cent[k][0], cent[k][1]
        r2 = (xx - xc) ** 2 + (yy - yc) ** 2
        matrix[:, :, k] = pupil * np.exp(-r2 / (0.75 * delta) ** 2)

    mat_flat = matrix[pupil]

    return matrix, pupil, mat_flat

# ==================================================================================================================== #
#                                              PSF Model
# ==================================================================================================================== #

class PointSpreadFunction(object):
    """
    PointSpreadFunction is in charge of computing the PSF
    for a given set of Zernike coefficients
    """

    def __init__(self, model_matrix, pupil_mask):

        self.model_matrix = model_matrix
        self.pupil_mask = pupil_mask

    def compute_PSF(self, coef):

        phase = np.dot(self.model_matrix, coef)
        pupil_function = self.pupil_mask * np.exp(1j * phase)
        image = (np.abs(fftshift(fft2(pupil_function))))**2

        return image

if __name__ == "__main__":

    """ Test Code to make sure PyCUDA is running properly """
    a = np.random.randn(4, 4).astype(np.float32)
    a_gpu = cuda.mem_alloc(a.nbytes)
    cuda.memcpy_htod(a_gpu, a)

    mod = SourceModule("""
        __global__ void doublify(float *a){
        int idx = threadIdx.x + threadIdx.y*4;
        a[idx] *= 2;}
        """)

    func = mod.get_function("doublify")
    func(a_gpu, block=(4, 4, 1))

    a_doubled = np.empty_like(a)
    cuda.memcpy_dtoh(a_doubled, a_gpu)

    # ================================================================================================================ #
    #                       CPU version
    # ================================================================================================================ #

    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)

    N_actuators = 45
    centers, MAX_FREQ = actuator_centres(N_actuators)
    # centers = radial_grid(N_radial=5)
    N_act = len(centers[0])
    plot_actuators(centers)

    rbf_mat = rbf_matrix(centers)        # Actuator matrix

    c_act = np.random.uniform(-1, 1, size=N_act)
    phase0 = np.dot(rbf_mat[0], c_act)
    p0 = min(phase0.min(), -phase0.max())

    plt.figure()
    plt.imshow(phase0, extent=(-1,1,-1,1), cmap='bwr')
    plt.colorbar()
    plt.clim(p0, -p0)
    for c in centers[0]:
        plt.scatter(c[0], c[1], color='black', s=4)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.show()

    PSF = PointSpreadFunction(rbf_mat[0], rbf_mat[1])

    ### Time how long it takes
    N_examples = 5000
    coefs = np.random.uniform(low=-1, high=1, size=(N_act, N_examples)).astype(np.float32)
    start = time.time()
    for i in range(N_examples):
        _img = PSF.compute_PSF(coefs[:, i])
    end = time.time()
    time_cpu = end - start
    print('\nCPU: %d images (%d, %d) pixels in %.3f seconds' %(N_examples, N_PIX, N_PIX, time_cpu))
    print('Average time: %.3f sec / example' %(time_cpu/N_examples))

    # ================================================================================================================ #

    start = time.time()
    phase_cpu = np.dot(rbf_mat[-1], coefs)
    end_dot = time.time()
    datacube_cpu = invert_mask_datacube(phase_cpu, PSF.pupil_mask)
    end = time.time()
    time_cpu = end_dot - start
    print("\nTime to produce %d Wavefronts [%d Actuators] in the CPU: %.3f sec" %(N_examples, N_act, time_cpu))

    # ================================================================================================================ #
    #                       GPU version
    # ================================================================================================================ #
    start_gpu = time.time()
    mod = SourceModule("""   
    __global__ void MatMulNoShared(float* A, float* B, float* C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols) {
        
        float CValue = 0;
        int TILE_DIM = blockDim.x;
        int Row = blockIdx.y*TILE_DIM + threadIdx.y;
        int Col = blockIdx.x*TILE_DIM + threadIdx.x;
    
        for (int k = 0; k < (TILE_DIM + ACols - 1)/TILE_DIM; k++) {  
            for (int n = 0; n < TILE_DIM; ++n) 
                if ((k*TILE_DIM + n < ACols && Row < ARows) && (k*TILE_DIM + n < BRows && Col < BCols))
                    CValue += A[Row*ACols + k*TILE_DIM + n] * B[(k*TILE_DIM + n)*BCols + Col];            
        }
        
        if (Row < CRows && Col < CCols) C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols)+(blockIdx.x*blockDim.x)+threadIdx.x]=CValue;
    }""")
    start_gpu = time.time()
    mod_shared = SourceModule("""
    #define TILE_DIM 32
    __global__ void MatMulShared(float* A, float* B, float* C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols) {
        float CValue = 0;
        //int TILE_DIM = blockDim.x;
        int Row = blockIdx.y*TILE_DIM + threadIdx.y;
        int Col = blockIdx.x*TILE_DIM + threadIdx.x;
        
        __shared__ float As[TILE_DIM][TILE_DIM];
        __shared__ float Bs[TILE_DIM][TILE_DIM];
        for (int k = 0; k < (TILE_DIM + ACols - 1)/TILE_DIM; k++) {
            if (k*TILE_DIM + threadIdx.x < ACols && Row < ARows)	As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*TILE_DIM + threadIdx.x];
            else													As[threadIdx.y][threadIdx.x] = 0.0;
            
            if (k*TILE_DIM + threadIdx.y < BRows && Col < BCols)	Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*BCols + Col];
            else													Bs[threadIdx.y][threadIdx.x] = 0.0;
            
            __syncthreads();
            for (int n = 0; n < TILE_DIM; ++n) CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];
            __syncthreads();
        }
        if (Row < CRows && Col < CCols) C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols)+(blockIdx.x*blockDim.x)+threadIdx.x]=CValue;
    }
    """)

    # func = mod.get_function("MatMulNoShared")
    func = mod_shared.get_function("MatMulShared")

    flat_actuator = rbf_mat[-1].copy().astype(np.float32)
    Nx = np.int32(flat_actuator.shape[0])   # N_PIX^2
    Ny = np.int32(N_act)
    Nz = np.int32(N_examples)
    BLOCK_SIZE = int(32)
    GRID_X = int(np.ceil(N_examples / BLOCK_SIZE))
    GRID_Y = int(np.ceil(flat_actuator.shape[0] / BLOCK_SIZE))

    # New_Phase [N_PIX^2, N_EXAMP] = Flat_Actuator [N_PIX^2, N_ACT] * Coefs [N_ACT, N_EXAMP]
    new_phase = np.zeros((Nx, Nz)).astype(np.float32)

    # Allocata GPU Memory
    flat_actuator_gpu = cuda.mem_alloc(flat_actuator.nbytes)
    new_phase_gpu = cuda.mem_alloc(new_phase.nbytes)
    coefs_gpu = cuda.mem_alloc(coefs.nbytes)

    # Transfer to GPU
    cuda.memcpy_htod(flat_actuator_gpu, flat_actuator)
    cuda.memcpy_htod(new_phase_gpu, new_phase)
    cuda.memcpy_htod(coefs_gpu, coefs)


    func(flat_actuator_gpu, coefs_gpu, new_phase_gpu, Nx, Ny, Ny, Nz, Nx, Nz, block=(BLOCK_SIZE, BLOCK_SIZE, 1), grid=(GRID_X, GRID_Y))
    cuda.memcpy_dtoh(new_phase, new_phase_gpu)
    end_dot_gpu = time.time()
    #
    # plt.imshow(new_phase - np.dot(flat_actuator, coefs))
    # plt.colorbar()
    # plt.show()

    # ================================================================================================================ #

    mod2 = SourceModule("""   
    __global__ void InvertMask(float* FlatPhase, float* PhaseCube, int* i_index, int* j_index, int N_PIX, int N_flat){

        int n_example = blockIdx.y;
        int z_flat = blockIdx.x * blockDim.x + threadIdx.x;
        float value;
        if (z_flat < N_flat) {
            value = FlatPhase[gridDim.y*z_flat + n_example];
            
            int i = i_index[z_flat];
            int j = j_index[z_flat];
            //if (n_example == 0) printf("\\n%d", i);
            
            PhaseCube[N_PIX * N_PIX * n_example + N_PIX * j + i] = value;
            //PhaseCube[i][j][n_example] = value;
            }  
        }""")


    phase_cube = np.zeros((N_examples, N_PIX, N_PIX)).astype(np.float32)
    i0 = np.linspace(0, N_PIX, N_PIX).astype(np.int32)
    ii, jj = np.meshgrid(i0, i0)
    i_f = ii[PSF.pupil_mask].astype(np.int32)
    j_f = jj[PSF.pupil_mask].astype(np.int32)

    phase_cube_gpu = cuda.mem_alloc(phase_cube.nbytes)
    i_gpu = cuda.mem_alloc(i_f.nbytes)
    j_gpu = cuda.mem_alloc(j_f.nbytes)

    cuda.memcpy_htod(phase_cube_gpu, phase_cube)
    cuda.memcpy_htod(i_gpu, i_f)
    cuda.memcpy_htod(j_gpu, j_f)

    func = mod2.get_function("InvertMask")
    GRID_X = int(np.ceil(flat_actuator.shape[0] / 1024))
    func(new_phase_gpu, phase_cube_gpu, i_gpu, j_gpu, np.int32(N_PIX), Nx, block=(1024, 1, 1), grid=(GRID_X, 5000))
    cuda.memcpy_dtoh(phase_cube, phase_cube_gpu)
    end_gpu = time.time()
    time_gpu = end_gpu - start_gpu
    print("\nTime to produce %d Wavefronts [%d Actuators] in the GPU: %.3f sec" % (N_examples, N_act, time_gpu))

    k = 0
    # p0 = invert_mask(new_phase[:, k], PSF.pupil_mask)
    p0 = invert_mask_datacube(new_phase, PSF.pupil_mask)[k]
    p1 = phase_cube[k]
    plt.figure()
    plt.imshow(p0)
    plt.figure()
    plt.imshow(p1)
    plt.show()

    mod3 = SourceModule("""
    #include <cufft.h>
    __global__ void FFT2D(float* FlatPhase, int N_PIX){
        cufftHandle plan;

        cufftPlan2d(&plan, N_PIX, N_PIX, CUFFT_C2C);
        
        }""")

    func = mod3.get_function("FFT2D")

    import pyculib.fft as pyfft



    # Use Curandom to generate the coefficients directly on the GPU
    from pycuda.curandom import rand as curand
    a_gpu = curand((50,))



    # ==================



    mod = SourceModule("""
        #include <stdio.h>

        __global__ void say_hi()
        {
          printf("I am %dth thread in threadIdx.x:%d.threadIdx.y:%d  blockIdx.:%d blockIdx.y:%d blockDim.x:%d blockDim.y:%d\\n",(threadIdx.x+threadIdx.y*blockDim.x+(blockIdx.x*blockDim.x*blockDim.y)+(blockIdx.y*blockDim.x*blockDim.y)),threadIdx.x, threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
        }
        """)

    func = mod.get_function("say_hi")
    func(block=(4, 4, 1), grid=(2, 2, 1))




