import cv2	
import sys
import imutils
import dlib
import numpy as np
from filters import face_filter, whiten_teeth
from imutils import face_utils

def main (argV = None):
	# Get user supplied values
	imagePath = sys.argv[1]
	faceFilterFactor = sys.argv[2]
	whiteningFactor = sys.argv[3]
	whiteningFactor = float(whiteningFactor)/100
	faceFilterFactor = float(faceFilterFactor)/100

	# Read the image
	image = cv2.imread(imagePath)
	# Convert to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	#Initialise face detector
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
	#Detect faces in grayed image
	rects = detector(gray,1)

	originalImage = np.copy(image)
	overlay = np.copy(image)

	#----------------------------------------Teeth Whitening-------------------------------------------------------------------------------

	#Loop through all faces
	for (i, rect) in enumerate(rects):

		#Create overlay
		overlay[:] = [0]

		#Determine Facial landmarks
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		#Create mask of mouth region
		(x,y) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
		pts = shape [x:y]
		hull = cv2.convexHull(pts)
		cv2.drawContours(overlay, [hull], -1, (255,255,255), -1)
		
		#Bitwise and with original image
		overlay = cv2.bitwise_and(overlay, image)

		#Whiten teeth
		teeth_mask = whiten_teeth(overlay, whiteningFactor)

		#Add image based on weight
		image [:,:,0] = (image [:,:,0] * (1 - whiteningFactor)) + (teeth_mask[:,:,0]* whiteningFactor *teeth_mask[:,:,3]) + (image [:,:,0] * whiteningFactor) - (image [:,:,0] * whiteningFactor * teeth_mask[:,:,3])
		image [:,:,1] = (image [:,:,1] * (1 - whiteningFactor)) + (teeth_mask[:,:,1]* whiteningFactor *teeth_mask[:,:,3]) + (image [:,:,1] * whiteningFactor) - (image [:,:,1] * whiteningFactor * teeth_mask[:,:,3])
		image [:,:,2] = (image [:,:,2] * (1 - whiteningFactor)) + (teeth_mask[:,:,2]* whiteningFactor *teeth_mask[:,:,3]) + (image [:,:,2] * whiteningFactor) - (image [:,:,2] * whiteningFactor * teeth_mask[:,:,3])
	
	#Writes teeth whitened image
	#cv2.imwrite('./outImages/teeth_whitening_only.jpg', image)

	#----------------------------------------Face Filter-------------------------------------------------------------------------------

	#Apply face filter
	#onlyFaceFilter = face_filter(originalImage)

	#Add image based on weight
	#onlyFaceFilter = cv2.addWeighted(originalImage, (1-faceFilterFactor), onlyFaceFilter, faceFilterFactor, 0)
	#Write image
	#cv2.imwrite('./outImages/face_filter_only.jpg', onlyFaceFilter)

	#Add image based on weight (Face filter + teeth whitening)
	teeth_whitened = np.copy(image)
	image = face_filter(image)
	image = cv2.addWeighted(teeth_whitened, (1-faceFilterFactor), image, faceFilterFactor, 0)
	#Write image
	path = "".join(("./outImages/", imagePath.split('/')[-1]))
	print (path)
	cv2.imwrite(path, image)



if __name__ == "__main__":    
    sys.exit(main())