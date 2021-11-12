import time
import os
import numpy as np
import random
import math
import scipy
import scipy.sparse

import functools
import multiprocessing


from math import ceil
import subprocess

from oasis.oasis_methods import oasisAR1, oasisAR2
from oasis.functions import gen_data, gen_sinusoidal_data, deconvolve, estimate_parameters


import torch



#Repo-specific imports
from masknmf.utils.computational_utils import l2_normalize
from masknmf.utils.computational_utils import dim_1_matmul
from masknmf.utils.computational_utils import runpar



   
    



#############
####
#### Low-Rank, memory efficient implementation of bessel init
####
#############

def get_projection_factorized_multiplier(U_sparse):
    '''
    Assumed U is a sparse coo matrix here
    Calculates ((U^t U)^-1)U^t
    '''
    print("get_projection_factorized multiplier")
    start_time = time.time()
    
    print("transpose found at {}".format(time.time() - start_time))


    print("finding value 1")
    value1 = U_sparse.T.tocsr()
    print("finding value 1 done at {}".format(time.time() - start_time))
    
    print("finding value 2")
    value2 = U_sparse.tocsr()
   
    print("{} us the fraction of all values of U are nonzero here".format(value2.count_nonzero()/ (value2.shape[0] * value2.shape[1])))
    print("finding value 2 done at {}".format(time.time() - start_time))
    
    print("finding prod now")
    prod = (value1 * value2).toarray()
    print("finding prod which is value1 * value2 done at {}".format(time.time() - start_time))
    
    print("prod found at {}".format(time.time() - start_time))
    prod = np.linalg.inv(prod)
    print("the shape of prod after np linalg is {}".format(prod.shape))
    print("the type of prod after np linalg is {}".format(type(prod)))
    print("inverse found at {}".format(time.time() - start_time))
    final_prod_t = U_sparse.tocsr().dot(prod.T)
    print("final_prod_ found at {}".format(time.time() - start_time))
    final_prod = final_prod_t.T  
    print("final_prod found at {}".format(time.time() - start_time))
    
    print("get_projection_factorized_multiplier took {}".format(time.time() - start_time))
    return final_prod


def deconv_trace(trace):
    _, s, _, _, _ = deconvolve(trace, penalty=1);
    return s


def get_factorized_projection(U_sparse, V, batch_size = 10000, device = 'cuda'):
    start_time = time.time()
    left_factor = get_projection_factorized_multiplier(U_sparse).astype("double")
    print("left factor found at {}".format(time.time() - start_time))
    print("the type of left facor is {}".format(left_factor.dtype))
    num_pixels = U_sparse.shape[0]
    
    accumulator = np.zeros_like(V)
    
    num_iters = math.ceil(U_sparse.shape[0]/batch_size)
    U_sparse = U_sparse.tocsr()
    print("entering for loop at time {}".format(time.time() - start_time))
    for i in range(num_iters):
        range_start = batch_size*(i)
        range_end = range_start + batch_size
        
        mov_portion = U_sparse[range_start:range_end, :].dot(V)
        print("the type of mov_portion is {}".format(type(mov_portion)))
        print("the shape of mov_portion is {}".format(mov_portion.shape))
        deconv_mov = np.array(runpar(deconv_trace, mov_portion)).astype("double");
        print("deconv finished at {}".format(time.time() - start_time))
        
       
        accumulator += dim_1_matmul(left_factor[:, range_start:range_end], deconv_mov, device = device,\
                                   batch_size = batch_size)
        print("iter {} done at {}".format(i, time.time() - start_time))
        
    return accumulator

    
def segment_local_UV(U, X, dims, obj_detector, frame_num, plot_mnmf = True, block_size = (16,16), order="F"):
    """
    Segments neurons using mask_model
    U: scipy.sparse.coo matrix, dimensions (d x R)
    X: np.ndarray, dimensions (R x T).
        NOTE: the product UX provides us with the projection video
    model: The mask-rcnn neural network filepath
    frame_num: The number of "brightest" frames we will observe over each region
    confidence: The threshold for accepting mask-rcnn components
    plot_mnmf: (boolean) Indicates whether we plot output as it is generated
    block_size: dims b1 x b2. The size of the block we use to partition the FOV. Over each block, we find the brightest frames, etc. 
    """

    
    time_mask_one= time.time()
    ## KEY HYPERPARAMETER
    

    mask_time = time.time()
    
    x, y, z = dims
    ## We perform an ordering step here: 
    frames = []
    portion = [] 
    iters1 = math.ceil(x / block_size[0])
    iters2 = math.ceil(y / block_size[1])
    
    ref_mat = np.arange(x*y)
    ref_mat_r = ref_mat.reshape((x,y),order=order)
    frame_array = np.zeros((iters1, iters2, frame_num))
    U = U.tocsr()

    for k in range(iters1):
        for j in range(iters2):
            x_range = (block_size[0] * k, block_size[0] * (k+1))
            y_range = (block_size[1] * j, block_size[1] * (j+1))
            
            index_values = ref_mat_r[x_range[0]:x_range[1], y_range[0]:y_range[1]].flatten()
            curr_U = U[index_values, :]
            curr_vid = curr_U.dot(X)

            brightness = np.amax(curr_vid, axis = 0)
            reorder = np.argsort(brightness)[::-1]
            max_values = reorder[:frame_num]
            
            frame_array[k, j, :] = max_values.flatten()
     
    
       
    ## We do not want to visit each frame multiple times. 
    ## For each frame, we create a list of tuples (a,b) describing regions over the FOV which are locally bright at this frame
    bright_dict = dict()
    x_indices,y_indices,z_indices = np.where(frame_array > -1)
    total_frame_vals = frame_array[(x_indices,y_indices,z_indices)]
    
    for k in range(len(total_frame_vals)):
        curr_frame_indexed = total_frame_vals[k]
        if curr_frame_indexed in bright_dict:
            bright_dict[curr_frame_indexed].append((x_indices[k], y_indices[k]))
        else:
            bright_dict[curr_frame_indexed] = [(x_indices[k], y_indices[k])]
    
    print("bright regions identified. starting segmentation")

    
    outputs = scipy.sparse.csc_matrix((x*y, 0))
    total_footprints = scipy.sparse.csc_matrix((x*y, 0))
    frame_numbers = []
    
    for key in bright_dict:
        tuple_list = bright_dict[key]
        frame_val = int(key)
        
        #Now we populate a frame of "valid positions":
        valid_positions = np.zeros((x,y))
        for index in range(len(tuple_list)):
            curr_tuple = tuple_list[index]
            x_range = (block_size[0] * curr_tuple[0], block_size[0] * (curr_tuple[0] + 1))
            y_range = (block_size[1] * curr_tuple[1], block_size[1] * (curr_tuple[1] + 1))
            valid_positions[x_range[0]:x_range[1], y_range[0]:y_range[1]] = 1
        
        curr_frame = U.dot(X[:, [frame_val]])
        detect_input = (curr_frame).reshape((x,y), order=order)
        masks = obj_detector.detect_instances(detect_input)
        footprints = masks.tocsr().multiply(curr_frame)
        footprints = footprints.tocsc()
        
        
        #Ignore components which are not in the bright regions
        valid_masks_indicator = masks.multiply(valid_positions.reshape((-1, 1))).tocsc()
        valid_masks_sum = valid_masks_indicator.sum(0)
        valid_masks_sum = np.asarray(valid_masks_sum)
        valid_masks_keep = (valid_masks_sum > 0).astype('bool')
        masks = masks[:, np.squeeze(valid_masks_keep)]
        footprints = footprints[:, np.squeeze(valid_masks_keep)]
        
        # print("the shape of outputs originally was {}".format(outputs.shape))
        # print("the shape of masks currently is {}".format(masks.shape))
        outputs = scipy.sparse.hstack([outputs, masks])
        # print("after appending masks hstack, the shape is {}".format(outputs.shape))
        total_footprints = scipy.sparse.hstack([total_footprints, footprints])
        for k in range(masks.shape[1]):
            frame_numbers.append(frame_val)
     
    
    ##Create a properties matrix: 
    property_count = 0
    properties = np.zeros((x,y))
    for k in range(iters1):
        for j in range(iters2):
            x_range = (block_size[0] * k, block_size[0] * (k+1))
            y_range = (block_size[1] * j, block_size[1] * (j+1))
            
            properties[x_range[0]:x_range[1], y_range[0]:y_range[1]] = property_count
            property_count +=1 
            
    properties_r = properties.reshape((-1, 1), order=order)  
    print("the shape of outputs at this point is {}".format(outputs.shape))
    neuron_properties = np.zeros((iters1*iters2, outputs.shape[1]))
    prod = outputs.multiply(properties_r)
    prod = prod.tocsc()
    
    for k in range(prod.shape[1]):
        curr_col = prod[:, k]
        row_nonzero, _ = curr_col.nonzero()
        property_values = np.unique(properties_r[row_nonzero, :]).astype('int')
        neuron_properties[property_values, k] = 1
                                
                                
    print("mask-rcnn segmentation took {}".format(time.time() - mask_time))
    return outputs, total_footprints, neuron_properties, frame_numbers                      


def filter_components_UV(masks, bin_masks, properties, mask_thres, bin_mask_thres):
    '''
    Inputs: 
        - masks: (d1*d2, K)-sized scipy.sparse.csc array. Describes K masks
        - bin_masks: (d1*d2, K)-sized scipy.sparse.csc array. Describes K binary masks
        - bound_boxes: np.ndarray. Dimensions (4 x K). For each of the k masks, provides a bounding box. bound_box[:, i] returns a column. First elt is min_x, second elt is min_y, third elt is max_x and fourth elt is max_y
        - properties: (P, K)-sized boolean ndarray. categories[i, j] is 1 if the j-th mask has property 'i'. 
        dims: tuple (d1, d2) describing the dimensions of the field of view
    '''
    
    mask_dims, num_masks = masks.shape
    
    ##Normalize masks data structure
    masks = l2_normalize(masks)
    bin_masks = l2_normalize(bin_masks)
    
    
    ##Normalize bin_masks data structure
    
    #Initialize dictionaries containing accepted components
    masks_dict = dict()
    bin_masks_dict = dict()
    keep_masks = np.zeros((num_masks, ), dtype='bool') #Tells us which masks we keep
    
    for property_val in range(properties.shape[0]):
        masks_dict[property_val] = scipy.sparse.csc_matrix((mask_dims, 1))
        bin_masks_dict[property_val] = scipy.sparse.csc_matrix((mask_dims, 1))
        
    for mask_index in range(num_masks):
        #Tells us whether we accept this mask
        curr_mask = masks[:, mask_index]
        curr_bin_mask = bin_masks[:, mask_index]
        
        ##Step 1: Check if it is highly similar to currently accepted real masks
        found_real_match_mask = False
        for property_val in range(properties.shape[0]):
            if properties[property_val, mask_index] == 0:
                continue
            else:
                accepted_masks = masks_dict[property_val]
                product = (curr_mask.T).dot(accepted_masks)
                max_similarity = product.max()
                if max_similarity > mask_thres:
                    found_real_match_mask = True
                
        
        ##Step 2: Check if it is highly similar to currently accepted bin masks
        found_bin_match_mask = False
        for property_val in range(properties.shape[0]):
            if properties[property_val, mask_index] == 0:
                continue
            else:
                accepted_masks = bin_masks_dict[property_val]
                product = (curr_bin_mask.T).dot(accepted_masks)
                max_similarity = product.max()
                if max_similarity > bin_mask_thres:
                    found_bin_mask_match = True
    
        if found_real_match_mask or found_bin_match_mask:
            continue
        else:
            keep_masks[mask_index] = 1
            
            #Also add this mask to the dictionaries: 
            for property_val in range(properties.shape[0]):
                if properties[property_val, mask_index] == 0:
                    continue
                else:
                    accepted_masks = masks_dict[property_val]
                    masks_dict[property_val] = scipy.sparse.hstack([accepted_masks, curr_mask])
                    
                    accepted_bin_masks = bin_masks_dict[property_val]
                    bin_masks_dict[property_val] = scipy.sparse.hstack([accepted_bin_masks, curr_bin_mask])
    
    return keep_masks
    



    
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
