import numpy as np
import torch
import scipy.sparse
from masknmf.detection.detector import ObjectDetector
import os
import multiprocessing

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.export.flatten import TracingAdapter
from detectron2.export.flatten import flatten_to_tuple
# from TracingAdapter import flatten_to_tuple

from masknmf.utils.image_transform import scale_to_RGB
import time

def inference_func(model, image):
    inputs = [{"image": image}]
    return model.inference(inputs, do_postprocess=False)[0]

class maskrcnn_detector():
    
    def __init__(self, net_path, config_path, confidence_level, allowed_overlap, cpu_only = False, order = "F"):
        '''
        Init function constructs the mask-rcnn network for object detection
        Params:
            net_path: string. describes the filepath of the neural network .pth file
            comfig_path: string. describes the filepath of neural network config.yaml file
            confidence_level: float between 0 and 1. the minimum confidence level for any segmentation provided by mask-rcnn. If an estimate below this confidence level, it is not considered. 
        '''
        self.predictor = self._initialize_predictor(net_path, config_path, confidence_level, cpu_only=cpu_only)
        self.allowed_overlap = allowed_overlap
        self.order = order
    
    def detect_instances(self, frame):
        '''
        Runs mask r-cnn detection on all frames of 'data'. Returns segmentation masks
        Params: 
            data: np.ndarray
        Returns: 
            masks_cropped. scipy.sparse.csc matrix, dimensions (d1*d2, K). d1, d2 are the dimensions of the FOV. K is the number of masks. 
        '''
        masks_sparse_list = self._get_masks_from_frame(frame)
        
        #Get rid of masks which overlap significantly 
        masks_sparse_list_prune = []
        for k in range(len(masks_sparse_list)):
            masks_sparse = masks_sparse_list[k]
            if masks_sparse.shape[1] > 1:
                #Elt (i, j) gives us total overlapping pixels b/w i-th and j-th masks
                elt_dot_prod = masks_sparse.T.dot(masks_sparse) 

                #Disregad dot products b/w a mask with itself
                elt_dot_prod.setdiag(0) 
                max_values = elt_dot_prod.max(axis = 1).toarray()
                max_values_thres = (max_values < self.allowed_overlap).astype('bool')

                masks_sparse_list_prune.append(masks_sparse[:, np.squeeze(max_values_thres)])

            else:
                masks_sparse_list_prune.append(masks_sparse)
        return masks_sparse_list_prune
        
    def _initialize_predictor_old(self, net_path, config_path, confidence_level, cpu_only=False):
        
        cfg = get_cfg()
        cfg.merge_from_file(config_path)
        cfg.MODEL.WEIGHTS = os.path.join(net_path) 
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_level 
        if cpu_only:
            cfg.MODEL.DEVICE='cpu'
        predictor = DefaultPredictor(cfg)
        return predictor
    
    def _initialize_predictor(self, net_path, config_path, confidence_level, cpu_only=False):
        cfg = get_cfg()
        cfg.merge_from_file(config_path)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_level # set threshold for this model
        if cpu_only:
            cfg.MODEL.DEVICE='cpu'   
        model = build_model(cfg) # returns a torch.nn.Module
        DetectionCheckpointer(model).load(net_path) 
        model.train(False) 
        
        model.eval()
        
        return model

#     def _get_masks_from_frame_batch(self, frames):
#         '''
#         Input: np.ndarray, dimensions (d1,d2, fr) where 'fr' is number frames
        
#         Output: List of masks in scipy.sparse.csc format (d1*d2, K) where K is number of masks
        
#         '''
#         img_list = []
#         print("starting to add all elements to data structure")
#         start_time = time.time()
#         for k in range(frames.shape[2]):
#             frame_RGB = scale_to_RGB(frames[:, :, k])
#             frame_RGB = np.transpose(frame_RGB,(2,0,1))
#             img_tensor = torch.from_numpy(frame_RGB).float()
            
#             img_list.append({"image": img_tensor})
            
#         print("elements added at time {}".format(time.time() - start_time))
        
#         print("running batch process inference")
#         start_time = time.time()
#         with torch.no_grad():
#             torch.set_num_threads(multiprocessing.cpu_count())
#             outputs = self.predictor(img_list)
#         print("batch process took {}".format(time.time() - start_time))
        
#         sparse_out = scipy.sparse.csc_matrix((frames.shape[0]*frames.shape[1], 0))
        
#         for k in range(frames.shape[2]):
#             instance_values = outputs[k]['instances']
#             if len(instance_values) > 0:
#                 pred_masks = instance_values.pred_masks
#                 values = pred_masks.cpu().detach().numpy().transpose(1,2,0)
#                 values_r = values.reshape((np.prod(values.shape[:2]),-1), order=self.order)
#                 values_sparse = scipy.sparse.csr_matrix(values_r).tocsc()

#                 #Get rid of components which overlap significantly
#                 prod_mat = values_sparse.T.dot(values_sparse)
#                 prod_mat.setdiag(0)
#                 max_vals = prod_mat.max(0).toarray()
#                 keep_elts = np.squeeze(max_vals < self.allowed_overlap)

#                 print("the time taken to filter the overlapping inputs is {}".format(time.time() - masks_time))

#                 sparse_out = scipy.sparse.hstack([sparse_out, values_sparse[:, keep_elts]])
#         return sparse_out
            

    
    def _get_masks_from_frame(self, frame):
        '''
        Frame: (d1, d2, N)-shape ndarray
        
        Outputs: 
            List of masks in scipy.sparse.csc format (d1*d2, K) where K is number of masks
        '''
        masks_time = time.time()
        img_list = []
        for k in range(frame.shape[2]):
            
            frame_RGB = scale_to_RGB(frame[:, :, k])
            frame_RGB = np.transpose(frame_RGB,(2,0,1))
            img_tensor = torch.from_numpy(frame_RGB).float()
            val = len(os.sched_getaffinity(os.getpid()))
            img_list.append({"image":img_tensor})

        
        masks_time = time.time()
        
        masks_time = time.time()
        with torch.no_grad():
            torch.set_num_threads(multiprocessing.cpu_count())
            outputs = self.predictor(img_list)
        print("the time taken to run maskrcnn on {} frames is {}".format(len(img_list), time.time() - masks_time))
        
        
        outputs_sparse = []
        masks_time = time.time()
        for k in range(len(outputs)):
            instance_values = outputs[k]['instances']
            if len(instance_values) > 0:
                pred_masks = instance_values.pred_masks
                values = pred_masks.cpu().detach().numpy().transpose(1,2,0)
                values_r = values.reshape((np.prod(values.shape[:2]),-1), order=self.order)
                values_sparse = scipy.sparse.csr_matrix(values_r).tocsc()

                #Get rid of components which overlap significantly
                prod_mat = values_sparse.T.dot(values_sparse)
                prod_mat.setdiag(0)
                max_vals = prod_mat.max(0).toarray()
                keep_elts = np.squeeze(max_vals < self.allowed_overlap)

                outputs_sparse.append(values_sparse[:, keep_elts])

            else:
                outputs_sparse.append(scipy.sparse.csc_matrix((frame.shape[0]*frame.shape[1], 0)))
        
    
        return outputs_sparse
    
    
    
    def _get_masks_from_frame_single(self, frame):
        '''
        
        Outputs: 
            List of masks in scipy.sparse.csc format (d1*d2, K) where K is number of masks
        '''
        masks_time = time.time()
        frame_RGB = scale_to_RGB(frame)
#         print("scale to RGB finished at point {}".format(time.time() - masks_time))
        frame_RGB = np.transpose(frame_RGB,(2,0,1))
#         print("transpose finished at point {}".format(time.time() - masks_time))
        img_tensor = torch.from_numpy(frame_RGB).float()
#         print("numpy to torch cast finished at {}".format(time.time() - masks_time))
        val = len(os.sched_getaffinity(os.getpid()))
#         print("before any mod, the os affinity is {}".format(val))

        img_list = [{"image":img_tensor}]
#         print("the time taken to scale to RGB + cast is {}".format(time.time() - masks_time))
        
        masks_time = time.time()
        
        masks_time = time.time()
        with torch.no_grad():
            torch.set_num_threads(multiprocessing.cpu_count())
            outputs = self.predictor(img_list)
        print("the time taken to run maskrcnn on 100 is {}".format(time.time() - masks_time))
        
        masks_time = time.time()
        instance_values = outputs[0]['instances']
        if len(instance_values) > 0:
            pred_masks = instance_values.pred_masks
            values = pred_masks.cpu().detach().numpy().transpose(1,2,0)
            values_r = values.reshape((np.prod(values.shape[:2]),-1), order=self.order)
            values_sparse = scipy.sparse.csr_matrix(values_r).tocsc()

            #Get rid of components which overlap significantly
            prod_mat = values_sparse.T.dot(values_sparse)
            prod_mat.setdiag(0)
            max_vals = prod_mat.max(0).toarray()
            keep_elts = np.squeeze(max_vals < self.allowed_overlap)
            
#             print("the time taken to filter the overlapping inputs is {}".format(time.time() - masks_time))
        
            return values_sparse[:, keep_elts]
        
        else:
            return scipy.sparse.csc_matrix((frame.shape[0]*frame.shape[1], 0))
        
