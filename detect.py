import numpy as np
import cv2 as cv
import model
import data
import matplotlib.pyplot as plt


face_classifier = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
source = cv.VideoCapture(0)
color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}
RED = (255, 255, 255)


def main():
    while True:
        ret, frame = source.read()
        frame = cv.flip(frame, 1, 1)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)
        for x, y, w, h in faces:
            face_img = frame[y:y + h, x:x + w]
            face_img = cv.cvtColor(face_img, cv.COLOR_BGR2RGB)
            resized = cv.resize(face_img, (data.IMG_SIZE, data.IMG_SIZE))
            reshaped = resized.reshape(1, data.IMG_SIZE, data.IMG_SIZE) / 255
            result = model.model.predict(reshaped)
            index = np.argmax(result[0])
            cv.rectangle(frame, (x, y), (x + w, y + h), color_dict[index], 2)
            cv.rectangle(frame, (x, y - 40), (x + w, y), color_dict[index], -1)
            cv.putText(frame, model.LABELS[index], (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)

        cv.imshow("LIVE", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
