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




    
def bessel_init_local_UV(U_sparse, R, V, dims, block_dims, frame_len, spatial_thres, model, \
                        plot_mnmf = True, batch_size = 10000, order="F"):
    
    '''
    Low-rank, memory efficient implementation of masknmf detection algorithm
    Params:
        U: scipy.sparse.coo_matrix: dimensions (d, R) where d is the number of pixels in the movie
        
        V: np.ndarray (2d): R x T Compressed Temporal Matrix from PMD
        block_dims: We partition the FOV into dimensions equal to block_dims when we want to find 'bright' components of the movie.
        frame_len: integer
    
    '''   
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    X = get_factorized_projection(U_sparse, R, V, batch_size = batch_size, device=device)
    
    
    bin_masks, footprints, properties, frame_numbers = segment_local_UV(U_sparse, R, X, dims, model, frame_len, plot_mnmf = plot_mnmf,\
                            block_size = block_dims, order=order)
    
    
    keep_masks = filter_components_UV(footprints, bin_masks, properties, spatial_thres[0], spatial_thres[1])
    
    
    final_masks = footprints[:, keep_masks]
    return final_masks
