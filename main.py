import cv2
from model import FacialExpressionModel
import numpy as np

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX


class VideoCamera(object):

    video = cv2.VideoCapture(0)

    while True:
        _, fr = video.read()
        fr_flipped = cv2.flip(fr,1)
        gray_fr = cv2.cvtColor(fr_flipped, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y + h, x:x + w]

            roi = cv2.resize(fc, (48, 48))

            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(fr_flipped, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr_flipped, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # have 4 images = real cam with canny prediction + real cam without canny prediction

        cv2.imshow("Emotion - Detector", fr_flipped)
        k = cv2.waitKey(1)
        if  k == 27:
            break

    video.release()
    cv2.destroyAllWindows()
