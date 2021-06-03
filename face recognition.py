import numpy as np
import cv2
new_path='C:/Users/inam.qadir/AppData/Local/Continuum/anaconda3/Library/etc/haarcascades/'
face_cascade = cv2.CascadeClassifier(new_path + 'haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier(new_path+'haarcascade_eye.xml')

img = cv2.imread('m10_dfh_ac.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('image',img)
print(np.shape(gray))
faces = face_cascade.detectMultiScale(img, 1.3, 5)

for (x,y,w,h) in faces:
	print(x,y,w,h)
face=img[y:y+w, x:x+h]
print(np.shape(face))
cv2.imwrite('crop.jpg', face)

cv2.imshow('img',face)

cv2.waitKey(0)
cv2.destroyAllWindows()