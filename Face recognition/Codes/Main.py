from imutils import paths
import face_recognition
import pickle
import cv2
import os

def train_model():
    path = '/home/pi/project/project/Face-recognition-based-attendance-system-master/known face photos/'
    # Our images are located in the dataset folder
    print("[INFO] Start processing faces...")
    imagePaths = list(paths.list_images(path))
    
    # Initialize the list of known encodings and known names
    knownEncodings = []
    knownNames = []
    
    # Loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # Extract the person name from the image path
        print("[INFO] Processing image {}/{}".format(i + 1, len(imagePaths)))
        name = imagePath.split(os.path.sep)[-1]
        
        # Load the input image and convert it from RGB (OpenCV ordering) to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect the (x, y)-coordinates of the bounding boxes corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb, model="hog")
        
        # Compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)
        
        # Loop over the encodings
        for encoding in encodings:
            # Remove file extension from name and add each encoding + name to our set of known names and encodings
            name = name[:-4]
            knownEncodings.append(encoding)
            knownNames.append(name)
    
    # Dump the facial encodings + names to disk
    print("[INFO] Serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    print(knownNames)
    with open("encodings.pickle", "wb") as f:
        f.write(pickle.dumps(data))
    
    return knownNames

names = []
names = train_model()
print("enroll : new registration")
print("attendance : take attendance")
print("exit : exit the program")

option = input('Select an operation: ')
while option != 'exit':
    if option == 'enroll':
        name = input('Enter name: ')
        import enroll, emailing
        enroll.enroll_via_camera(name)
        names = train_model()
    elif option == 'attendance':
        import recognition, spreadsheet
        spreadsheet.mark_all_absent()
        currentname = recognition.run_recognition()
        print(currentname)
        spreadsheet.write_to_sheet(currentname)
    
    option = input('Select an option: ')
