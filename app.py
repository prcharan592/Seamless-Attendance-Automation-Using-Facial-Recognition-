# Importing necessary libraries
import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

# Initializing Flask app
app = Flask(__name__)

# Getting today's date in two formats for file naming and display
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Initializing OpenCV face detector with Haar Cascade
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Ensuring required directories exist
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
# Creating a CSV file for today's attendance if it doesn't exist
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

# Function to count total registered users
def totalreg():
    return len(os.listdir('static/faces'))

# Function to extract faces from an image
def extract_faces(img):
    try:
        if img.shape != (0, 0, 0):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for better detection
            face_points = face_detector.detectMultiScale(gray, 1.3, 5)  # Detect faces
            return face_points
        else:
            return []
    except:
        return []

# Function to identify a face using the pre-trained model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')  # Load saved model
    return model.predict(facearray)

# Function to train the KNN model on user face data
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))  # Resize face images for consistency
            faces.append(resized_face.ravel())  # Flatten the image array
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)  # Initialize KNN classifier
    knn.fit(faces, labels)  # Train the model
    joblib.dump(knn, 'static/face_recognition_model.pkl')  # Save the trained model

# Function to extract attendance records from the CSV file
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

# Function to add attendance for a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')

# Function to get all registered users
def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)
    
    return userlist, names, rolls, l

# Function to delete a user's folder
def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(duser + '/' + i)  # Remove all images
    os.rmdir(duser)  # Remove the folder

################ ROUTING FUNCTIONS ################

# Home route - Displays attendance records
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# Start Attendance route - Captures real-time attendance
@app.route('/start', methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)  # Draw rectangle around detected face
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))  # Extract and resize the face
            identified_person = identify_face(face.reshape(1, -1))[0]  # Predict the person's identity
            add_attendance(identified_person)  # Log their attendance
            cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:  # Exit loop on pressing 'Esc'
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# Add User route - Registers a new user
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 10 == 0:
                name = newusername + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == 500:  # Capture 50 images
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()  # Retrain model after adding a new user
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# Running the Flask app
if __name__ == '__main__':
    app.run(debug=True)
