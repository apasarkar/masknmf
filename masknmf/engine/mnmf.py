import numpy as np

import scipy
import scipy.sparse



#Repo-specific imports
from masknmf.engine.sparsify import get_factorized_projection
from masknmf.engine.segmentation import segment_local_UV, filter_components_UV



#############
####
#### Low-Rank, memory efficient implementation of bessel init
####
#############




    
def bessel_init_local_UV(U_sparse, V, dims, block_dims, frame_len, spatial_thres, model, \
                        plot_mnmf = True, device = 'cuda', batch_size = 10000, order="F"):
    
    '''
    Low-rank, memory efficient implementation of masknmf detection algorithm
    Params:
        U: scipy.sparse.coo_matrix: dimensions (d, R) where d is the number of pixels in the movie
        V: np.ndarray (2d): R x T Compressed Temporal Matrix from PMD
        block_dims: 2-element tuple
            The block size used in PMD denoising. Elements must be divisible by 4
        frame_len: integer
    
    '''    
    X = get_factorized_projection(U_sparse, V, batch_size = batch_size, device = device)
    
    
    bin_masks, footprints, properties, frame_numbers = segment_local_UV(U_sparse, X, dims, model, frame_len, plot_mnmf = plot_mnmf,\
                            block_size = block_dims, order=order)
    
    
    keep_masks = filter_components_UV(footprints, bin_masks, properties, spatial_thres[0], spatial_thres[1])
    
    
    final_masks = footprints[:, keep_masks]
    return final_masks
