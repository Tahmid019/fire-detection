import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import xception


#masking function
def create_mask_for_plant(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_hsv = np.array([0,0,250])
    upper_hsv = np.array([250,255,255])
    
    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

#image segmentation function
def segment_image(image):
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask = mask)
    return output/255

#sharpen the image
def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp

def read_img(img):
    #convert image to array
    img = image.img_to_array(img)
    return img


def xception_preprocess_input(img):
    
    img = read_img(img)
    
    #masking and segmentation
    image_segmented = segment_image(img)
    
    #sharpen
    image_sharpen = sharpen_image(image_segmented)
    
    x = xception.preprocess_input(np.expand_dims(image_sharpen.copy(), axis=0))
    return x
    
def xception_bg(img):
    x = xception_preprocess_input(img)
    
    xception_bf = xception.Xception(weights='imagenet', include_top=False, pooling='avg')
    bf_train_val = xception_bf.predict(x, batch_size=1, verbose=1)
    
    return bf_train_val


