import cv2

cap = cv2.VideoCapture(0)


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    if len(faces) == 0:
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y + w, x:x + h], faces[0]


i = 0
while True:
    ret, frame = cap.read()

    img = frame.copy()
    face, rect = detect_face(img)

    if rect is not None:
        draw_rectangle(img, rect)
        i += 1
        name = "new-face/" + str(i) + '.jpg'
        cv2.imwrite(name, frame)
        print(i, ".jpg saved")

    cv2.imshow("Web Camera", img)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        i += 1
        name = "new-face/"+str(i) + '.jpg'
        cv2.imwrite(name, frame)
        print(i, ".jpg saved")
    # elif cv2.waitKey(0) & 0xFF == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()