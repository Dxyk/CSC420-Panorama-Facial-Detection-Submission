import cv2
import matplotlib.pyplot as plt
import numpy as np
from model_train_eval import evaluate, actor_names

# Load the cascade
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
# img = cv2.imread('bill.jpg')
img = cv2.imread('./Resource/breaking_bad_test.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

new_faces = []
# for face in faces:

for i in [1, 2, 3, 6]:
    face = faces[i]
    x, y, w, h = face
    this_face = img[y:y + h, x:x + w]
    new_faces.append(this_face)
    cv2.imwrite("face{}.jpg".format(len(new_faces)),
                cv2.cvtColor(this_face, cv2.COLOR_RGB2BGR))

f, ax = plt.subplots(nrows=len(new_faces), ncols=1)
for i in range(len(new_faces)):
    ax[i].imshow(new_faces[i], cmap="gray")
img_cpy = np.copy(img)

# for i, face in enumerate(faces):
for i in [1, 2, 3, 6]:
    face = faces[i]
    x, y, w, h = face
    tl = (x, y)
    tl1 = (x, y + 10)
    tl2 = (x, y + 50)
    br = (x + w, y + h)
    img_cpy = cv2.rectangle(img_cpy, tl, br, (0, 255, 0), 3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # output from the
    name, gender = evaluate(img[y:y + h, x:x + w])
    img_cpy = cv2.putText(img_cpy, name, tl1, font, 1.5, (0, 255, 0), 2,
                          cv2.LINE_AA)
    img_cpy = cv2.putText(img_cpy, gender, tl2, font, 1.5, (0, 255, 0), 2,
                          cv2.LINE_AA)
plt.imshow(img_cpy)
cv2.imwrite("failure.jpg", cv2.cvtColor(img_cpy, cv2.COLOR_RGB2BGR))

cv2.imwrite("bb_test_1.jpg", cv2.cvtColor(new_faces[1], cv2.COLOR_BGR2RGB))
cv2.imwrite("bb_test_2.jpg", cv2.cvtColor(new_faces[2], cv2.COLOR_BGR2RGB))
cv2.imwrite("bb_test_3.jpg", cv2.cvtColor(new_faces[3], cv2.COLOR_BGR2RGB))
cv2.imwrite("bb_test_4.jpg", cv2.cvtColor(new_faces[6], cv2.COLOR_BGR2RGB))
