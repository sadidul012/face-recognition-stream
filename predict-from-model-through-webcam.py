import cv2

subjects = ["Unknown", "Sadidul Islam"]
face_recognizer = cv2.face.createLBPHFaceRecognizer(threshold=60)
face_recognizer.load("face_recognizer_LBPH.yml")


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    if len(faces) == 0:
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y + w, x:x + h], faces[0]


def predict(test_img):
    img = test_img.copy()
    face, rect = detect_face(img)
    not_found = True
    if face is not None:
        label = face_recognizer.predict(face)

        if label is None:
            label_text = subjects[0]
        else:
            label_text = subjects[label]
            print(label_text)
            not_found = True
        draw_rectangle(img, rect)
        draw_text(img, label_text, rect[0], rect[1] - 5)
    return img, not_found

#
# print("Predicting images...")
#
# test_img3 = cv2.imread("11.jpg")
# test_img4 = cv2.imread("test-data/test4.jpg")
#
# predicted_img3, flag = predict(test_img3)
# predicted_img4, flag = predict(test_img4)
# print("Prediction complete")
# # print(predicted_img3)
# cv2.imshow(subjects[1], cv2.resize(predicted_img3, (400, 500)))
# cv2.imshow(subjects[0], cv2.resize(predicted_img4, (400, 500)))
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.waitKey(1)
# cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
continuous = True
frame = None
while True:
    if continuous:
        ret, frame = cap.read()
        face, rect = detect_face(frame)

        if face is not None:
            cv2.imwrite("test.jpg", frame)
            frame, not_found = predict(frame)
            continuous = not_found

    cv2.imshow('Face Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
