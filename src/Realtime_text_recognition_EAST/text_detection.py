from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2
import pytesseract


cap = cv2.VideoCapture(0) # '0' for laptop cam '1' for phone camera

net = cv2.dnn.readNet("frozen_east_text_detection.pb")


while True:
	ret, image = cap.read()
	orig = image
	(H, W) = image.shape[:2]

	(newW, newH) = (640, 320)
	rW = W / float(newW)
	rH = H / float(newH)

	image = cv2.resize(image, (newW, newH))
	try:
		(H, W) = image.shape[:2]
	except:
		pass

	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]


	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)

	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)

	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	for y in range(0, numRows):

		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		
		for x in range(0, numCols):
			
			if scoresData[x] < 0.5:
				continue

			
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	boxes = non_max_suppression(np.array(rects), probs=confidences)

	for (startX, startY, endX, endY) in boxes:

		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)
		boundary = 2

		text = orig[startY-boundary:endY+boundary, startX - boundary:endX + boundary]
		text = cv2.cvtColor(text.astype(np.uint8), cv2.COLOR_BGR2GRAY)
		textRecongized = pytesseract.image_to_string(text)
		cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 3)
		cv2.putText(orig, textRecongized, (endX,endY+5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
		print(textRecongized)
		
	cv2.imshow("Text Detection", orig)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break
cv2.destroyAllWindows()
cap.release()