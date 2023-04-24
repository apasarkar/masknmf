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

import torch
import torch_sparse

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
  # trace = trace - jnp.amin(trace)
  trace = jnp.clip(trace, a_min=0, a_max=None)
  trace = trace - jnp.percentile(trace, thres_val)
  trace = jnp.clip(trace, a_min=0, a_max=None)

  return trace


# @partial(jit)
def oasis_deconv_ar1(trace, lambda_val, gamma_val):

  trace = preprocess_data(trace)
  s_init = jnp.zeros_like(trace) 
  solver = ProjectedGradient(fun=objective_function,
                             projection=projection.projection_non_negative,
                             tol=1e-6, maxiter=100)
  fit_val = solver.run(s_init, trace=trace, lambda_val=lambda_val, gamma_val=gamma_val).params

  return fit_val


# oasis_deconv_ar1_vmap = jit(vmap(oasis_deconv_ar1, in_axes=(0, None, None)), static_argnums=(1,2))

oasis_deconv_ar1_vmap = vmap(oasis_deconv_ar1, in_axes=(0, None, None))



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

@partial(jit, static_argnums=(2,3))
def fused_deconvolution_pipeline(UR, V_crop, lambda_val, gamma_val):
    mov_portion = jnp.matmul(UR, V_crop)
    return oasis_deconv_ar1_vmap(mov_portion, lambda_val, gamma_val)

#This function batches over frames
def get_factorized_projection_jax(U_sparse, R, V, batch_size = 1000, frame_upper_bound=10000, lambda_val = 0.7, gamma_val = 0.95):
    
    num_iters = math.ceil(U_sparse.shape[0]/batch_size)
    UR = U_sparse.tocsr().dot(R)
    X = np.zeros((U_sparse.shape[0],min(frame_upper_bound, V.shape[1])))
    index_points = list(np.arange(0, U_sparse.shape[0] - batch_size, batch_size))
    index_points.append(U_sparse.shape[0] - batch_size)
    V_temporal_crop = V[:, :frame_upper_bound]
    for i in tqdm(index_points):
        range_start = i
        range_end = min(range_start + batch_size, U_sparse.shape[0])
        
        deconv_mov = fused_deconvolution_pipeline(UR[range_start:range_end, :], V_temporal_crop, lambda_val, gamma_val)

        X[range_start:range_end, :] = np.array(deconv_mov).astype(np.float32)
    
    UTX = U_sparse.transpose().dot(X)
    final_prod = R.T.dot(UTX) 
    return final_prod


def build_mat(g, T, device):
    '''
    Builds the weighted deconvolution matrix for "deconvolving the ar-1 process with parameter "g"
    '''
    rows_1 = torch.arange(T, device=device)
    rows_2 = torch.arange(T-1, device=device)
    rows = torch.hstack([rows_1, rows_2])
    
    columns_1 = torch.arange(T, device=device)
    columns_2 = torch.arange(1, T, device=device)
    columns = torch.hstack([columns_1, columns_2])
    
    values_1 = torch.ones(T, device=device)
    values_2 = torch.ones(T-1, device=device)*-g
    values = torch.hstack([values_1, values_2])
    
    values[0] = 0 #Zero out the first frame since deconvolution is not meaningful here
    deconv_mat = torch_sparse.tensor.SparseTensor(row=rows, col=columns, value=values, sparse_sizes = (T, T))
    
    return deconv_mat
    
    
def get_factorized_projection(U_sparse, R, V, batch_size = 1000, percentile=.95, device='cpu'):
    deconv_mat = build_mat(0.98, V.shape[1], device)
    V = torch.Tensor(V).float().to(device)
    original_X = torch_sparse.matmul(deconv_mat.t(), V.t()).t() #Dimensions R x T
    
    cumulator = torch.zeros_like(original_X, device=device)
    
    num_iters = math.ceil(U_sparse.shape[0]/batch_size)
    
    U_sparse_torch = torch_sparse.tensor.from_scipy(U_sparse).float().to(device)
    R_torch = torch.Tensor(R).float().to(device)
    threshold_object = torch.nn.ReLU(inplace=True)
    for k in range(num_iters):
        start_pixel = k*batch_size
        end_pixel = min((k+1)*batch_size, U_sparse.shape[0])
        
        indices = torch.arange(start_pixel, end_pixel, device=device)
        U_sparse_torch_selected = torch_sparse.index_select(U_sparse_torch, 0, indices)
        UR_crop = torch_sparse.matmul(U_sparse_torch_selected, R_torch)
        
        URX_crop = torch.matmul(UR_crop, original_X)
        threshold_object(URX_crop)
        quantiles = torch.quantile(URX_crop, percentile, dim=1, keepdim=True)
        URX_crop.subtract_(quantiles)
        threshold_object(URX_crop)
        
        cumulator.add_(torch.matmul(UR_crop.t(), URX_crop))
        
        
    return cumulator.cpu().numpy()
        
        
        
        
        


    
