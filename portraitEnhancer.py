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

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
	rects = detector(gray,1)
	overlay = np.copy(image)
	overlay[:] = [0]

	originalImage = np.copy(image)

	for (i, rect) in enumerate(rects):
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		for (name, (i,j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
			if name == "mouth":
				(x,y) = face_utils.FACIAL_LANDMARKS_IDXS[name]
				pts = shape [x:y]
				hull = cv2.convexHull(pts)
				cv2.drawContours(overlay, [hull], -1, (255,255,255), -1)
			
	overlay = cv2.bitwise_and(overlay, image)

	whiten_teeth_image = whiten_teeth(overlay, whiteningFactor)
	
	image [:,:,0] = (image [:,:,0] * 0.5) + (whiten_teeth_image[:,:,0]* 0.5 *whiten_teeth_image[:,:,3]) + (image [:,:,0] * 0.5) - (image [:,:,0] * 0.5 * whiten_teeth_image[:,:,3])
	image [:,:,1] = (image [:,:,1] * 0.5) + (whiten_teeth_image[:,:,1]* 0.5 *whiten_teeth_image[:,:,3]) + (image [:,:,1] * 0.5) - (image [:,:,1] * 0.5 * whiten_teeth_image[:,:,3])
	image [:,:,2] = (image [:,:,2] * 0.5) + (whiten_teeth_image[:,:,2]* 0.5 *whiten_teeth_image[:,:,3]) + (image [:,:,2] * 0.5) - (image [:,:,2] * 0.5 * whiten_teeth_image[:,:,3])
	#cv2.imshow("Only Teeth Whitening",image)
	cv2.imwrite('./outImages/teeth_whitening_only.jpg', image)

	onlyFaceFilter = face_filter(originalImage)
	onlyFaceFilter = cv2.addWeighted(originalImage, (1-faceFilterFactor), onlyFaceFilter, faceFilterFactor, 0)
	cv2.imwrite('./outImages/face_filter_only.jpg', onlyFaceFilter)

	image = cv2.addWeighted(image, (1-faceFilterFactor), onlyFaceFilter, faceFilterFactor, 0)
	#cv2.imshow("Teeth Whitening + Face Filter", image)
	cv2.imwrite('./outImages/teeth_whitening_and_face_filter.jpg', image)



if __name__ == "__main__":    
    sys.exit(main())