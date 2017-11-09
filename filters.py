import skimage
import cv2
import numpy as np
from skimage import io, filters
from PIL import ImageEnhance


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
    boundaries = [
    ([150,170,170],[213,226,222])
    ]


    for lower, upper in boundaries:

        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        mask = cv2.inRange(image, lower, upper)
        alpha_channel = np.copy(mask)
        alpha_channel [alpha_channel == 255] = 1

        mask = cv2.bilateralFilter(mask, 15, 75, 75)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        b,g,r = cv2.split(mask)
        mask = cv2.merge((b,g,r,alpha_channel))
        alpha = 1 - whiteningFactor

        image [:,:,0] = (image [:,:,0] * alpha) + (mask[:,:,0]*(1 - alpha)*mask[:,:,3]) + (image [:,:,0] * (1-alpha)) - (image [:,:,0] * (1-alpha) * mask[:,:,3])
        image [:,:,1] = (image [:,:,1] * alpha) + (mask[:,:,1]*(1 - alpha)*mask[:,:,3]) + (image [:,:,1] * (1-alpha)) - (image [:,:,1] * (1-alpha) * mask[:,:,3])
        image [:,:,2] = (image [:,:,2] * alpha) + (mask[:,:,2]*(1 - alpha)*mask[:,:,3]) + (image [:,:,2] * (1-alpha)) - (image [:,:,2] * (1-alpha) * mask[:,:,3])
        

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


def blur_filter(image, sigma):
    """ Gaussian blurring.
    Args
    ----
        sigma: higher value, more radical blurring (sigma > 0)
    """
    blurred = filters.gaussian(image, sigma=sigma, multichannel=True)
    return blurred


def sharpen_filter(image, a, sigma):
    """ Sharpening an image.
    Args
    ----
        a: how strongly the edge constrast effect is applied (a > 0)
        sigma: lower value, finer edges that will be detected (sigma > 0)
    """
    blurred = blur_filter(image, sigma)
    sharper = np.clip(image * (1.0 + a) - blurred * a, 0, 1.0)
    return sharper