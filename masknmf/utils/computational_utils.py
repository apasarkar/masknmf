import numpy as np
import scipy.sparse



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
    sum_cols[sum_cols < 0.0000000001] = 1
    sum_cols_sqrt = np.sqrt(sum_cols)
    sum_cols_recip = np.reciprocal(sum_cols_sqrt)

    sum_cols_recip_mat = scipy.sparse.spdiags(sum_cols_recip, 0, scipy_csc_sq.shape[1], scipy_csc_sq.shape[1])
    scipy_csc_normalized = scipy_csc_mat.dot(sum_cols_recip_mat)

    
    return scipy_csc_normalized.tocsc()


    
    
    
    
