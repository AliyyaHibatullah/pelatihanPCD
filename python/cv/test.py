import cv2
import os
import os.path
import numpy as np
from time import sleep
import pickle
def read_model(filename, path=""):
    with open(os.path.join(path, filename), 'rb') as in_name:
        model = pickle.load(in_name)
        return model


car_cascade = cv2.CascadeClassifier('cars.xml')
font = cv2.QT_FONT_NORMAL
color = (255, 0, 0)
stroke = 2
offset=6 #error tiap piksel
from skimage.transform import resize

pos_line=100 #Posisi garis deteksi

delay= 100 #FPS video

detec = []
car = 0

def titik_pusat (x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

cap = cv2.VideoCapture('video2.mp4')
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
print(height)
print(width)

model = read_model("model2.pkl", path="") #load model
while True:
    ret , frame1 = cap.read()
    tempo = float(1/delay)
    
    sleep(tempo) 
    # grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(grey,(3,3),5)
    cars = car_cascade.detectMultiScale(frame1,  minNeighbors=5) 
    cv2.line(frame1, (25, pos_line), (1200, pos_line), (255,127,0), 3) 

    for (x, y, w, h) in cars:
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (255, 0, 0), (2)) #kotakan deteksi

        center = titik_pusat(x, y, w, h)
        detec.append(center)

        # cv2.circle(frame1, center, 4, (0, 0,255), -1)
        roi_gray = frame1[y:y + h, x:x + w]
        roi_color = frame1[y:y + h, x:x + w]
        convert = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
        muka=resize(roi_gray,(200,200,3)) #wajib sama dengan citra inputan trainer
        # muka = cv2.GaussianBlur(resized,(5,5),0)
          

        for (x,y) in detec:
            if y>(pos_line+offset):
                l=[muka.flatten()] #wajib di flatten
                # #model == model weight algoritma terbaik yg diload
                id_= int(model.predict(l)[0]) #prediksi
                if(id_==0):
                    car+=1
                    cv2.imshow("Detect",muka)
        detec.remove((x,y))
    cv2.putText(frame1, "VEHICLE COUNT : "+str(car), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)
    cv2.imshow("Video Original" , frame1)

    if cv2.waitKey(1) == 27:
        break
    
cv2.destroyAllWindows()
cap.release()
