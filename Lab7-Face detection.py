import cv2
from PIL import Image
from tkinter import filedialog

def detect_faces(image_path):
 
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    face_cascade_path = "haarcascade_frontalface_default.xml"  
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face Detection Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = filedialog.askopenfilename(initialdir="/", title="Select Image", filetypes=(("Image Files", "*.jpg;*.jpeg;*.png"),))

if image_path:
    detect_faces(image_path)


