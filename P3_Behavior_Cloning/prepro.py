#Image pre-processing
import numpy as np
from PIL import Image

def img_pre(load_img):
    my_size=160,80
    #resize
    test_img=load_img.resize(my_size, Image.ANTIALIAS)
    img_arr=np.asarray(test_img)
    #Normalization
    #image_array[:,:,:,0]=image_array[:,:,0]-106.83322356
    #image_array[:,:,:,1]=image_array[:,:,1]-109.46058697
    #image_array[:,:,:,2]=image_array[:,:,2]-102.59168762
    return img_arr
