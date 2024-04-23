import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

labels = ["I", "Am", "Happy", "Sad", "Hungry", "Need", "Help"]
detected_signs = []  # List to store detected signs

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            detected_sign = labels[index]  # Get detected sign

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            detected_sign = labels[index]  # Get detected sign

        # Add detected sign to the list if it's not already in the list
        if detected_sign not in detected_signs:
            detected_signs.append(detected_sign)

    # Draw all detected signs with reduced font size and word wrapping
    sign_text = ' '.join(detected_signs)
    font_scale = 0.5
    font_thickness = 1
    text_size, _ = cv2.getTextSize(sign_text, cv2.FONT_HERSHEY_COMPLEX, font_scale, font_thickness)
    text_width = text_size[0]
    if text_width > imgOutput.shape[1] - 40:  # Check if text exceeds image width
        detected_signs = []  # Reset detected signs list if text exceeds image width
        cv2.putText(imgOutput, "Signs Exceeded Image Width", (20, imgOutput.shape[0] - 20), cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 0, 255), font_thickness, lineType=cv2.LINE_AA)
    else:
        cv2.putText(imgOutput, sign_text, (20, imgOutput.shape[0] - 20), cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 0, 255), font_thickness, lineType=cv2.LINE_AA)

    cv2.imshow("Image", imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
