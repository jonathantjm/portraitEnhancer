# CS2108 Project
## Requirements

1. Python 2.7
1. OpenCV

## Installation

### OpenCV:
1. Download here : https://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.3.0/opencv-3.3.0-vc14.exe/download
1. Then follow these instructions : https://docs.opencv.org/3.1.0/d5/de5/tutorial_py_setup_in_windows.html
	1. If you're using Anaconda and created a separate environment, in step 6, you would copy it to `...Anaconda3/envs/py27/Lib`

### dlib
1. Download here : https://pypi.python.org/packages/76/44/d5ea0c2cfbbce3c12841de8e29ba97e305c585b98c70f8e8f1cb69855f1f/dlib-18.17.100-cp27-none-win32.whl
1. Run `pip install dlib-file.whl` in cmd

### imutils
1. Run `pip install imutils` in cmd

## How to run
In cmd, just run `python portraitEnhancer.py $$$ xx yy`, where $$$ is the name of the picture, xx is a number from 0-100 representing the intensity of **Face Filter** and yy is a number from 0-100 representing the intensity of **Teeth Whitening**

After running, three images will be written to outImages folder:
1. Image with teeth whitening
1. Image with face filter
1. Image with teeth whitening + face filter