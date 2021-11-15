import torch
import numpy as np
import scipy.sparse
import math
import scipy
import time
import functools
import multiprocessing
import os


def l2_normalize(scipy_csc_mat, tol = 0.00000001):
    '''
    Routine to do l2 normalization on every column of a scipy array
    Inputs: 
        scipy_csc_mat. scipy.sparse.csc_matrix. Dimensions (d1, d2).
    Outputs: 
        norm_mat. scipy.sparse.csc_matrix. Dimensions (d1, d2). Matrix with all columns normalized. 
    '''
    
    scipy_csc_sq = scipy_csc_mat.multiply(scipy_csc_mat)
    sum_cols = np.asarray(scipy_csc_sq.sum(0))
    sum_cols[sum_cols<tol] = 1
    sum_cols_sqrt = np.sqrt(sum_cols)
    sum_cols_recip = np.reciprocal(sum_cols_sqrt)
    
    scipy_csc_normalized = scipy_csc_sq.multiply(sum_cols_recip)
    
    return scipy_csc_normalized.tocsc()


def dim_1_matmul(A, B, device = 'cuda', batch_size = 10000):
    '''
    GPU-accelerated matmul of A x B and B x C matrix. Use this method when B is extremely large but A x C can fit on GPU
    '''
    if device == 'cpu':
        print("dim1matmul cpu")
        print("NEW NP DOT")
        print("the shape of A is {}".format(A.shape))
        print("the shape of B is {}".format(B.shape))
        print("the type of A is {}".format(A.dtype))
        print("the type of B is {}".format(B.dtype))
        val = len(os.sched_getaffinity(os.getpid()))
        print("the number of usable CPUs is {}".format(val))
        print("the num mkl threads is {}".format(os.environ['MKL_NUM_THREADS']))
#         print(os.environ)
        A_t = torch.from_numpy(A)
        B_t = torch.from_numpy(B)
        start_time = time.time()
        #with torch.no_grad():
        torch.set_num_threads(24)
        prod = torch.matmul(A_t, B_t)
        end_time = time.time() - start_time
        print("the matmul itself took {}".format(end_time))
        return prod.numpy()
        #return np.dot(A, B)
    
    accumulator = np.zeros((A.shape[0], B.shape[1]))
    
    batch_values = math.ceil((A.shape[1]/batch_size))
    for k in range(batch_values):
        interval_start = batch_size*k
        interval_end = batch_size*(k+1)
        A_t = torch.from_numpy(A[:, interval_start:interval_end]).to(device)
        B_t = torch.from_numpy(B[interval_start:interval_end, :]).to(device)
        out = torch.matmul(A_t, B_t).to('cpu').detach().numpy()
        accumulator += out
    torch.cuda.empty_cache()
    return accumulator


def runpar(f, X, nprocesses=None, **kwargs):
    '''
    res = runpar(function,          # function to execute
                 data,              # data to be passed to the function
                 nprocesses = None, # defaults to the number of cores on the machine
                 **kwargs)          # additional arguments passed to the function (dictionary)
    '''
    
    #Change affinity (if needed) to enable full multicore processing
    
    
    val = len(os.sched_getaffinity(os.getpid()))
    print("the CPU affinity BEFORE runpar is {}".format(val))

    if nprocesses is None:
        nprocesses = int(multiprocessing.cpu_count()) 
        print("the number of processes is {}".format(nprocesses))
#         val = len(os.sched_getaffinity(os.getpid()))
#         print('the number of usable cpu cores is {}'.format(val))
    
    with multiprocessing.Pool(initializer=parinit, processes=nprocesses) as pool:
        res = pool.map(functools.partial(f, **kwargs), X)
    pool.join()
    pool.close()


    val = len(os.sched_getaffinity(os.getpid()))
    print("after the multicore, the affinity is {}".format(val))

    num_cpu = multiprocessing.cpu_count()
    os.system('taskset -cp 0-%d %s' % (num_cpu, os.getpid()))
    val = len(os.sched_getaffinity(os.getpid()))
    print("the cpu affinity after the process (intro fix) is {}".format(val))
    return res

def parinit():
    import os
    os.environ['MKL_NUM_THREADS'] = "1"
    os.environ['OMP_NUM_THREADS'] = "1"
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    num_cpu = multiprocessing.cpu_count()
    os.system('taskset -cp 0-%d %s' % (num_cpu, os.getpid()))

    
    
    
    
