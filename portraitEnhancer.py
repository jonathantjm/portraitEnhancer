import cv2
import sys
import skimage
import numpy as np
from filters import face_filter, whiten_teeth

def main (argV = None):
	# Get user supplied values
	imagePath = sys.argv[1]
	faceCascPath = "haarcascade_frontalface_default.xml"
	smileCascPath = "haarcascade_smile.xml"

	# Create the haar cascade
	faceCascade = cv2.CascadeClassifier(faceCascPath)
	smileCascade = cv2.CascadeClassifier(smileCascPath)

	# Read the image
	image = cv2.imread(imagePath)
	# Convert to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	smile_x = 0
	smile_y = 0
	smile_w = 0
	smile_h = 0	
	face_x = 0
	face_y = 0
	face_w = 0
	face_h = 0

	# Detect face in the image
	sF = 1.01
	while 1:
		faces = faceCascade.detectMultiScale(
		    gray,
		    scaleFactor=sF,
		    minNeighbors=7,
		    minSize=(55, 55)
		)
		if len(faces) == 1:
			break
		if sF > 1.3:
			print("Sorry! A face could not be detected Try another image")
			sys.exit()
		sF = sF + 0.01
	#sF = 1.01
	for (x,y,w,h) in faces:
		face_x = x
		face_y = y
		face_w = w
		face_h = h
    
    # Detect smiles in the image
	sF = 1.2
	while 1:
		smiles = smileCascade.detectMultiScale(
		    gray,
		    scaleFactor=sF,
		    minNeighbors=8,
		    minSize=(15, 15)
		)
		if len(smiles) == 1 or sF > 1.8:
			break
		sF = sF + 0.1
	
	maxWhite = 0
	# Iterates through all smiles found and finds the one most likely to be a smile, based on percentage of white 
	for (x,y,w,h) in smiles:
		# Gets cropped image
		croppedImage = image[y:(y+h), x:(x+w)]
		# Converts to grayscale
		croppedImage = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2GRAY)
		# Get B&W histogram
		hist = cv2.calcHist([croppedImage],[0],None,[256],[0,256])
		# Calculates percentage of white
		whiteSum = 0;
		for i in range(199,255):
			whiteSum = whiteSum + hist[i]
		percentageWhite = whiteSum / (h*w)
		# Maintains max
		if percentageWhite > maxWhite:
			maxWhite = whiteSum
			smile_x = x
			smile_y = y
			smile_w = w
			smile_h = h

	
	onlyFaceFilter = np.copy(image)
	cv2.imshow("Before", image)
	whiten_teeth_image = whiten_teeth(image[smile_y:(smile_y+smile_h), smile_x:(smile_x+smile_w)])
	
	image[smile_y:(smile_y+smile_h), smile_x:(smile_x+smile_w)] = cv2.addWeighted(image[smile_y:(smile_y+smile_h), smile_x:(smile_x+smile_w)], 0.5, whiten_teeth_image, 0.5, 0)
	cv2.imshow("Only Teeth Whitening", image)
	#cv2.imshow("Teeth Whitening", whiten_teeth_image)
	#cv2.imshow("Whitened Teeth", whiten_teeth_image)	
	image = face_filter(image)
	onlyFaceFilter = face_filter(onlyFaceFilter)

	cv2.imshow("Only Face Filter", onlyFaceFilter)
	cv2.imshow("Teeth Whitening + Face Filter", image)
	cv2.waitKey(0)




if __name__ == "__main__":    
    sys.exit(main())