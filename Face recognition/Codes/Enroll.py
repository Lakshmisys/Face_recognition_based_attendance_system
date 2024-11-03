import face_recognition
import cv2
import pickle
import spreadsheet
from picamera import PiCamera
from picamera.array import PiRGBArray
import RPi.GPIO as GPIO
import time

photo_folder = '/home/pi/project/project/Face-recognition-based-attendance-system-master/known face photos/'
facial_encodings_folder = '/home/pi/project/project/Face-recognition-based-attendance-system-master/known_face_encodings/'

def encoding_of_enrolled_person(name, image):
    enroll_encoding = []
    enroll_encoding.append(face_recognition.face_encodings(face_recognition.load_image_file(image)))
    with open(facial_encodings_folder + name + '.txt', 'wb') as fp:
        pickle.dump(enroll_encoding, fp)

def enroll_via_camera(name):
    cam = PiCamera()
    cam.resolution = (512, 304)
    cam.framerate = 10
    rawCapture = PiRGBArray(cam, size=(512, 304))
    
    for frame in cam.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array
        cv2.imshow("Press Space to take a photo", image)
        rawCapture.truncate(0)
        k = cv2.waitKey(1)
        rawCapture.truncate(0)
        
        if k % 256 == 27:  # ESC pressed
            print('Quitting')
            cv2.destroyAllWindows()
            break
        elif k % 256 == 32:  # Space pressed
            img_name = photo_folder + name + '.jpg'
            cv2.imwrite(img_name, image)
            encoding_of_enrolled_person(name, photo_folder + name + '.jpg')
            cv2.destroyAllWindows()
            break
    
    email = input("Enter email address: ")
    spreadsheet.enroll_person_to_sheet(name, email)
    cv2.destroyAllWindows()
    cam.close()
