import numpy as np


def scale_to_RGB(img):
    '''
    Scale a (d1, d2)-shape image to a RGB value
    '''
    curr_shape = img.shape
    new_img = np.zeros((img.shape[0], img.shape[1], 3))
    if np.count_nonzero(img) == 0:
        print("the image is empty")
    else:
        normalizer = np.amax(img)
        gscale_img = img/normalizer
        scaled = gscale_img*255  #To get 255, multiply by 255 here
        scaled = np.clip(scaled, 0, 255)
        rgb_img = np.floor(scaled).astype('int') #To scale to 255, change this to 'int'
        
        new_img[:, :, 0] = rgb_img
        new_img[:, :, 1] = rgb_img
        new_img[:, :, 2] = rgb_img
        
    
        
    return new_img.astype(np.uint8)