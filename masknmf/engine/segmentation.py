import torch
import numpy as np
import time
import os
import random
import math

import scipy.sparse

from masknmf.utils.computational_utils import l2_normalize


def get_ordering_cuda(U, R, X, x, y, z, block_size, frame_num=20):
    iters1 = math.ceil(x / block_size[0])
    iters2 = math.ceil(y / block_size[1])

    ref_mat = np.arange(x*y)
    ref_mat_r = ref_mat.reshape((x,y),order='C')
    frame_array = torch.zeros((iters1, iters2, frame_num), device='cuda')

    UR = torch.from_numpy(U.tocsr().dot(R)).cuda()
    X_cuda = torch.from_numpy(X).cuda()
    ref_mat_torch = torch.from_numpy(ref_mat_r).cuda()
    frame_array = torch.zeros((iters1, iters2, frame_num), device='cuda')


    for k in range(iters1):
        for j in range(iters2):
            x_range = torch.tensor([block_size[0] * k, block_size[0] * (k+1)], device='cuda')
            y_range = torch.tensor([block_size[1] * j, block_size[1] * (j+1)], device='cuda')

            index_values = torch.flatten(ref_mat_torch[x_range[0]:x_range[1], y_range[0]:y_range[1]])
            UR_crop = UR[index_values, :]
            prod = torch.matmul(UR_crop, X_cuda)
            max_values = torch.max(prod, dim=0)[0]

            frame_array[k, j, :] = torch.argsort(max_values, descending=True)[:frame_num]

    return frame_array.detach().cpu().numpy()


    
def segment_local_UV(U, R, X, dims, obj_detector, frame_num, plot_mnmf = True, block_size = (16,16), order="F"):
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
    
    frame_array = get_ordering_cuda(U, R, X, x, y,z, block_size, frame_num)
    
       
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
    
    print("identifying bright regions took {}".format(time.time() - mask_time))
    print("bright regions identified. starting segmentation")

    print("now populating valid positions")
    val_pos_time = time.time()
    
    outputs = scipy.sparse.csc_matrix((x*y, 0))
    total_footprints = scipy.sparse.csc_matrix((x*y, 0))
    frame_numbers = []
    
    frame_examine_list = [key for key in bright_dict]
    batch_step = 50
    
    for key_ind in range(len(frame_examine_list)):
        if key_ind % batch_step == 0:
            #Run detectron on this next set of frames in batch: 
            frame_range = range(key_ind, min(len(frame_examine_list), key_ind + batch_step))
            frame_sublist = [int(frame_examine_list[i]) for i in frame_range]
            
            RX_curr = R.dot(X[:, frame_sublist])
            frame_to_examine = U.dot(RX_curr)
            detect_input = frame_to_examine.reshape((x,y, -1), order = order)
            masks_list = obj_detector.detect_instances(detect_input)
            
        dict_time = time.time()
        tuple_list = bright_dict[frame_examine_list[key_ind]]
        frame_val = int(frame_examine_list[key_ind])
        
        #Now we populate a frame of "valid positions":
        valid_positions = np.zeros((x,y))
        for index in range(len(tuple_list)):
            curr_tuple = tuple_list[index]
            x_range = (block_size[0] * curr_tuple[0], block_size[0] * (curr_tuple[0] + 1))
            y_range = (block_size[1] * curr_tuple[1], block_size[1] * (curr_tuple[1] + 1))
            valid_positions[x_range[0]:x_range[1], y_range[0]:y_range[1]] = 1
        
        

        curr_frame = frame_to_examine[:, [(key_ind % batch_step)]]
        masks = masks_list[(key_ind % batch_step)]
        footprints = masks.tocsr().multiply(curr_frame)
        footprints = footprints.tocsc()
        

        
        #Ignore components which are not in the bright regions
        valid_masks_indicator = masks.multiply(valid_positions.reshape((-1, 1), order=order)).tocsc()
        valid_masks_sum = valid_masks_indicator.sum(0)
        valid_masks_sum = np.asarray(valid_masks_sum)
        valid_masks_keep = (valid_masks_sum > 0).astype('bool')
        masks = masks[:, np.squeeze(valid_masks_keep)]
        footprints = footprints[:, np.squeeze(valid_masks_keep)]
        
#         print("reject components not in bright region finished at {}".format(time.time() - dict_time))
        
        # print("the shape of outputs originally was {}".format(outputs.shape))
        # print("the shape of masks currently is {}".format(masks.shape))
        outputs = scipy.sparse.hstack([outputs, masks])
        # print("after appending masks hstack, the shape is {}".format(outputs.shape))
        total_footprints = scipy.sparse.hstack([total_footprints, footprints])
        
#         print("hstack finish at {}".format(time.time() - dict_time))
        
        for k in range(masks.shape[1]):
            frame_numbers.append(frame_val)
        
        
#         print("one iteration out of {} iterations took {}".format(len(list(bright_dict.keys())),time.time() - dict_time))
     
    print("the detection step overall took {}".format(time.time() - val_pos_time))
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
                    break
                    
        if found_real_match_mask:
            continue
        
        
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
    
        if found_bin_match_mask:
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
    
    print("we started with {} masks and kept {} masks".format(keep_masks.shape, np.count_nonzero(keep_masks)))
    return keep_masks
    
