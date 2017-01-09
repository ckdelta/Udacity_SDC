#Image pre-processing
import numpy as np
from PIL import Image

def img_pre(load_img):
    my_size=160,80
    test_img=load_img.resize(my_size, Image.ANTIALIAS)
    img_arr=np.asarray(test_img)
    return img_arr
