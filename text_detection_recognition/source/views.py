from django.shortcuts import render, redirect
from django.http import HttpResponse
from imutils.object_detection import non_max_suppression
from text_detection_recognition.settings import BASE_DIR
import numpy as np
import time
import cv2
import pytesseract
# Create your views here.

def home(request):
    return render(request, 'source/home.html')

def textDetect(request):
    

    cap = cv2.VideoCapture(0) # '0' for laptop cam '1' for phone camera

    net = cv2.dnn.readNet(str(BASE_DIR) + "/source/frozen_east_text_detection.pb")


    while True:
        ret, image = cap.read()
        orig = image
        try:
            (H, W) = image.shape[:2]
        except:
            pass
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
    return redirect('/')

def textRecognize(request):
    
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    img = cv2.imread(str(BASE_DIR) + '/source/dataset/sample.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(pytesseract.image_to_string(img))


    #Detecting Characters
    def detect_characters(img):
        height, width, index = (img.shape)
        boxes = pytesseract.image_to_boxes(img)
        for b in boxes.splitlines():
            b =b.split(' ')
            #print(b)
            x,y,w,h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
            cv2.rectangle(img, (x,height-y), (w,height-h),(0,255,0), 3)
            cv2.putText(img, b[0], (x,height-y + 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

        cv2.imshow('Frame', img)
        cv2.waitKey(0)
        

    #detect_characters(img)
    def detect_words(img):
        height, width, index = (img.shape)
        boxes = pytesseract.image_to_data(img)
        for x,b in enumerate(boxes.splitlines()):
            if x!=0:
                b =b.split()
                #print(b)
                if len(b) == 12:
                    x,y,w,h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                    cv2.rectangle(img, (x,y), (w+x,h+y),(0,255,0), 3)
                    cv2.putText(img, b[11], (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

        cv2.imshow('Frame2', img)
        cv2.waitKey(0)
        
    detect_words(img)
    return redirect('/')