import face_recognition
import cv2
import os
import pickle
import numpy as np
import spreadsheet
from imutils.video import VideoStream, FPS
import imutils
import time

known_face_encodings = []
known_face_names = []

photo_folder = '/home/pi/project/project/Face-recognition-based-attendance-system-master/known face photos/'
facial_encodings_folder = '/home/pi/project/project/Face-recognition-based-attendance-system-master/known face encodings/'

def load_facial_encodings_and_names_from_memory():
    for filename in os.listdir(facial_encodings_folder):
        known_face_names.append(filename[:-4])
        with open(facial_encodings_folder + filename, 'rb') as fp:
            known_face_encodings.append(pickle.load(fp)[0])

def run_recognition():
    # Initialize 'currentname' to trigger only when a new person is identified.
    currentname = "unknown"
    # Determine faces from encodings.pickle file model created from train_model.py
    encodingsP = "encodings.pickle"
    
    # Load the known faces and embeddings along with OpenCV's Haar cascade for face detection
    print("[INFO] Loading encodings + face detector...")
    data = pickle.loads(open(encodingsP, "rb").read())
    
    # Initialize the video stream and allow the camera sensor to warm up
    vs = VideoStream(usePiCamera=True).start()
    time.sleep(2.0)
    
    # Start the FPS counter
    fps = FPS().start()
    
    # Loop over frames from the video file stream
    while True:
        # Grab the frame from the threaded video stream and resize it to 500px (to speed up processing)
        frame = vs.read()
        frame = imutils.resize(frame, width=500)
        
        # Detect the face boxes
        boxes = face_recognition.face_locations(frame)
        
        # Compute the facial embeddings for each face bounding box
        encodings = face_recognition.face_encodings(frame, boxes)
        names = []
        
        # Loop over the facial embeddings
        for encoding in encodings:
            # Attempt to match each face in the input image to our known encodings
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"  # if face is not recognized, then print Unknown
            
            # Check to see if we have found a match
            if True in matches:
                # Find the indexes of all matched faces and initialize a dictionary to count matches
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                
                # Loop over the matched indexes and maintain a count for each recognized face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                
                # Determine the recognized face with the largest number of votes
                name = max(counts, key=counts.get)
                
                # If someone in your dataset is identified, print their name on the screen
                if currentname != name:
                    currentname = name
                    print(currentname)
            
            # Update the list of names
            names.append(name)
        
        # Loop over the recognized faces and draw the predicted face name on the image
        for ((top, right, bottom, left), name) in zip(boxes, names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 225), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Display the image on the screen
        cv2.imshow("Facial Recognition is Running", frame)
        key = cv2.waitKey(1) & 0xFF
        
        # Quit when 'q' key is pressed
        if key == ord("q"):
            break
        
        # Update the FPS counter
        fps.update()
    
    # Stop the timer and display FPS information
    fps.stop()
    print("[INFO] Elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))
    
    # Cleanup
    cv2.destroyAllWindows()
    vs.stop()
    
    return currentname
