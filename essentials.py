
class essentials:
    subjects = []

    def __init__(self):
        self.subjects = ["Unknown", "Sadidul Islam"]

    def draw_rectangle(self, img, rect):
        (x, y, w, h) = rect
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def draw_text(self, img, text, x, y):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    def predict(self, test_img):
        img = test_img.copy()
        face, rect = detect_face(img)

        label = face_recognizer.predict(face)

        if label is None:
            label_text = subjects[0]
        else:
            label_text = subjects[label]
        draw_rectangle(img, rect)
        draw_text(img, label_text, rect[0], rect[1] - 5)
        return img