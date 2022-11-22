import numpy as np
import scipy.sparse
import time
import math
from tqdm import tqdm
from oasis.oasis_methods import oasisAR1, oasisAR2
from oasis.functions import gen_data, gen_sinusoidal_data, deconvolve, estimate_parameters


from functools import partial
import jax
from jax import jit, vmap
import jax.numpy as jnp
from jaxopt import projection
from jaxopt import ProjectedGradient
import jaxopt

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
    np.nan_to_num(s, copy=False, nan=0)
    return s

# @partial(jit)
def get_c_from_s(s, gamma):
  final_val = jnp.zeros_like(s)
  s_part1 = jax.lax.dynamic_slice(s, (0,), (jnp.size(s)-1,))
  s_part1 = jnp.insert(s_part1, jnp.array([0]), jnp.array([0]))

  return s + gamma*s_part1


# @partial(jit)
def objective_function(s, trace, lambda_val, gamma_val):
  c = get_c_from_s(s, gamma_val)
  norm1 = jnp.linalg.norm(c - trace) ** 2
  sum_val = jnp.sum(s)
  return norm1 + lambda_val*sum_val

# @partial(jit)
def preprocess_data(trace, thres_val=15):
  trace = trace - jnp.amin(trace)
  trace = trace - jnp.percentile(trace, thres_val)
  trace = jnp.clip(trace, a_min=0, a_max=None)

  return jax.nn.normalize(trace)


# @partial(jit)
def oasis_deconv_ar1(trace, lambda_val, gamma_val):

  trace = preprocess_data(trace)
  s_init = jnp.zeros_like(trace) 
  solver = ProjectedGradient(fun=objective_function,
                             projection=projection.projection_non_negative,
                             tol=1e-6, maxiter=100)
  fit_val = solver.run(s_init, trace=trace, lambda_val=lambda_val, gamma_val=gamma_val).params

  return fit_val


oasis_deconv_ar1_vmap = jit(vmap(oasis_deconv_ar1, in_axes=(0, None, None)), static_argnums=(1,2))



#This function batches over pixels - better to parallelize over frames since deconvolution is the limiting step here
def get_factorized_projection_pixels(U_sparse, R, V, batch_size = 10000, device = 'cuda'):
    
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

#This function batches over frames
def get_factorized_projection_multiprocess(U_sparse, R, V, batch_size = 1000, device = 'cuda'):
    
    num_iters = math.ceil(V.shape[1]/batch_size)
    print("the value of V shape is {}".format(V.shape))
    print("the value of batch size is {}".format(batch_size))
    UR = U_sparse.tocsr().dot(R)
    X = np.zeros_like(V)
    for i in tqdm(range(num_iters)):
        range_start = batch_size*(i)
        range_end = range_start + batch_size
        
        mov_portion = UR.dot(V[:, range_start:range_end])
        
        orig_type = mov_portion.dtype
        deconv_mov = np.array(runpar(deconv_trace, mov_portion.astype("float64"))).astype(orig_type);

        X[:, range_start:range_end] = (UR.T).dot(deconv_mov)
    return X


#This function batches over frames
def get_factorized_projection_old(U_sparse, R, V, batch_size = 1000, lambda_val = 0.7, gamma_val = 0.95, device = 'cuda'):
    
    num_iters = math.ceil(V.shape[1]/batch_size)
    print("the value of V shape is {}".format(V.shape))
    print("the value of batch size is {}".format(batch_size))
    UR = U_sparse.tocsr().dot(R)
    X = np.zeros_like(V)
    for i in tqdm(range(num_iters)):
        range_start = batch_size*(i)
        range_end = range_start + batch_size
        
        mov_portion = UR.dot(V[:, range_start:range_end])
        
        orig_type = mov_portion.dtype
        deconv_mov = oasis_deconv_ar1_vmap(mov_portion, lambda_val, gamma_val).astype(orig_type)

        X[:, range_start:range_end] = (UR.T).dot(deconv_mov)
    return X



#This function batches over frames
def get_factorized_projection(U_sparse, R, V, batch_size = 1000, frame_upper_bound=10000, lambda_val = 0.7, gamma_val = 0.95):
    
    num_iters = math.ceil(U_sparse.shape[0]/batch_size)
    UR = U_sparse.tocsr().dot(R)
    X = np.zeros((U_sparse.shape[0],min(frame_upper_bound, V.shape[1])))
    index_points = list(np.arange(0, U_sparse.shape[0] - batch_size, batch_size))
    index_points.append(U_sparse.shape[0] - batch_size)
    for i in tqdm(index_points):
        range_start = i
        range_end = min(range_start + batch_size, U_sparse.shape[0])
        
        mov_portion = UR[range_start:range_end, :].dot(V[:, :frame_upper_bound])
        
        orig_type = mov_portion.dtype
        deconv_mov = oasis_deconv_ar1_vmap(mov_portion, lambda_val, gamma_val).astype(orig_type)

        X[range_start:range_end, :] = deconv_mov
    
    final_prod = (UR.T).dot(X)
    return final_prod



    
