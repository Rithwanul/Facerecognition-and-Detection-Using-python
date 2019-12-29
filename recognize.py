import cv2

def generate_dataset(img, id, img_id):
    cv2.imwrite("data/user."+str(id)+"."+str(img_id)+".jpg", img)

def draw_boundary(img, classifier, scalefactor, minNeighbour, color, text, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scalefactor, minNeighbour)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        id, _ = clf.predict(gray_img[y:y+h, x:x+w])
        if id == 1:
            cv2.putText(img, "Mumin", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]

    return coords

def recognize_user(img, clf, facecascade):
    color = {"blue" : (255, 0, 0), "green" : (0, 255, 0), "red" : (0, 0, 255)}
    draw_boundary(img, facecascade, 1.1, 10, color["blue"], "Face", clf)
    return img;




faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")



video_capture = cv2.VideoCapture(0)
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")


img_id = 0

while True:
    _, img = video_capture.read()
    img = recognize_user(img, clf, faceCascade)
    cv2.imshow("Face Detection", img)
    img_id += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()