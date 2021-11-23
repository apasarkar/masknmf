import numpy as np
import scipy.sparse
import time
import math

from oasis.oasis_methods import oasisAR1, oasisAR2
from oasis.functions import gen_data, gen_sinusoidal_data, deconvolve, estimate_parameters

from masknmf.utils.multiprocess_utils import runpar



def get_projection_factorized_multiplier(U_sparse):
    '''
    Assumed U is a sparse coo matrix here
    Calculates ((U^t U)^-1)U^t
    '''

    value1 = U_sparse.T.tocsr()
    value2 = U_sparse.tocsr()
    
    prod = (value1 * value2).toarray()

    prod = np.linalg.inv(prod)
    final_prod_t = U_sparse.tocsr().dot(prod.T)
    final_prod = final_prod_t.T  
    
    return final_prod


def deconv_trace(trace):
    _, s, _, _, _ = deconvolve(trace, penalty=1);
    return s


def get_factorized_projection(U_sparse, R, V, batch_size = 10000, device = 'cuda'):
    
    start_time = time.time()
    num_pixels = U_sparse.shape[0]
    
    accumulator = np.zeros_like(V)
    
    num_iters = math.ceil(U_sparse.shape[0]/batch_size)
    U_sparse = U_sparse.tocsr()
    for i in range(num_iters):
        range_start = batch_size*(i)
        range_end = range_start + batch_size
        
        UR_crop = U_sparse[range_start:range_end, :].dot(R)
        mov_portion = UR_crop.dot(V)
        
        orig_type = mov_portion.dtype
        deconv_mov = np.array(runpar(deconv_trace, mov_portion.astype("float64"))).astype(orig_type);

        
        accumulator += (UR_crop.T).dot(deconv_mov)
        
    return accumulator

    