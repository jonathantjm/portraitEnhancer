import skimage
import cv2
import numpy as np
from skimage import io, filters


def face_filter(original_image):
    image = skimage.img_as_float(original_image)
    
    # midtone red contrast boost
    image = midtone_filter(image, 2, [
            0, 0.05, 0.1, 0.2, 0.3,
            0.5, 0.7, 0.8, 0.9, 0.95,
            1.0])
    
    # midtone blue contrast weaken
    image = midtone_filter(image, 0, [
            0, 0.047, 0.118, 0.251, 0.318,
            0.392, 0.42, 0.439, 0.475, 0.561,
            0.58, 0.627, 0.671, 0.733, 0.847,
            0.925, 1])
    
    image = skimage.img_as_ubyte(image)

    return image

def whiten_teeth (image, whiteningFactor):

    #Boundary of teeth colour
    boundaries = [([122,146,164],[214,245,255])]

    #Loop through all boundaries
    for lower, upper in boundaries:

        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        #Create mask which fulfill boundary criteria
        mask = cv2.inRange(image, lower, upper)

        #Create alpha channel based on mask
        alpha_channel = np.copy(mask)
        alpha_channel [alpha_channel == 255] = 1

        #Blur mask
        mask = cv2.bilateralFilter(mask, 15, 150, 150)

        #Merge alpha channel
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        b,g,r = cv2.split(mask)
        mask = cv2.merge((b,g,r,alpha_channel))
        alpha = 1 - whiteningFactor

        #Blend mask and original image based on weights and alpha channel
        image [:,:,0] = (image [:,:,0] * alpha) + (mask[:,:,0]*(1 - alpha)*mask[:,:,3]) + (image [:,:,0] * (1-alpha)) - (image [:,:,0] * (1-alpha) * mask[:,:,3])
        image [:,:,1] = (image [:,:,1] * alpha) + (mask[:,:,1]*(1 - alpha)*mask[:,:,3]) + (image [:,:,1] * (1-alpha)) - (image [:,:,1] * (1-alpha) * mask[:,:,3])
        image [:,:,2] = (image [:,:,2] * alpha) + (mask[:,:,2]*(1 - alpha)*mask[:,:,3]) + (image [:,:,2] * (1-alpha)) - (image [:,:,2] * (1-alpha) * mask[:,:,3])
        
    #Create alpha channel and merge with image
    alpha_channel = cv2.inRange(image, (0,0,0), (0,0,0))
    alpha_channel [alpha_channel == 0] = 1
    alpha_channel [alpha_channel == 255] = 0
    b,g,r = cv2.split(image)
    image = cv2.merge((b,g,r,alpha_channel))    
    return image

def midtone_filter(image, channels, values):
    """ Tone and color correction. Use linear interplotation
    to approximate the correction curve.
    Args
    ----
        channels: interger or a list of intergers, inidicating the 
                  channels that the correction applies
        values: the y-axis values that control the linear interplotation
    """
    working_image = image[:, :, channels]
    orig_size = working_image.shape
    flatten = working_image.flatten()
    # apply the linear interplotation to the pixels
    adjusted = np.interp(flatten, np.linspace(0, 1, len(values)), values)
    adjusted = adjusted.reshape(orig_size)
    ret_image = np.array(image)
    ret_image[:, :, channels] = adjusted
    return ret_image