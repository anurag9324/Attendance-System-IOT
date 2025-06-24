import pandas as pd
import cv2
import urllib.request
import numpy as np
import os
from datetime import datetime
import face_recognition
import time  # Added for sleep function

path = r'C:\Users\anura\OneDrive\Desktop\attendance\attendance\image_folder'
url = 'http://192.168.154.194/cam-hi.jpg'

# Get subject name from the user
subject_name = input("Enter the subject name: ")

# Get today's date in dd/mm/yy format
today = datetime.today().strftime('%d/%m/%y')

# Delete existing attendance.csv file
if 'attendance.csv' in os.listdir(os.path.join(os.getcwd())):
    os.remove('attendance.csv')
    print("Deleted existing attendance.csv file.")

# Create a new attendance.csv file
df = pd.DataFrame(columns=['Subject', 'Name', 'Login Time', 'Date', 'Status'])
df.to_csv("attendance.csv", index=False)
print("Created a new attendance.csv file.")

images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name, period, date):
    with open("attendance.csv", 'r+') as f:
        myDataList = f.readlines()

        # Check if the file is not empty
        if myDataList:
            nameList = [entry.split(',')[1] for entry in myDataList if len(entry.split(',')) >= 2]
        else:
            nameList = []

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{period},{name},{dtString},{date},Present')
            return "Marked present for {}".format(name)
        else:
            return "{} is already marked present".format(name)


encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Create the window with the flag cv2.WINDOW_NORMAL
cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Webcam', 800, 600)

message_display_duration = 5  # seconds
message_displayed = False  # Flag to track if the message has been displayed

while True:
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgnp, -1)
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            if not message_displayed:
                message = markAttendance(name, subject_name, today)
                print(message)

                # Display the message for a short duration
                cv2.putText(img, message, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                message_displayed = True
                start_time = time.time()

            # Check if the message display duration has elapsed
            elapsed_time = time.time() - start_time
            if elapsed_time > message_display_duration:
                # Clear the message after the duration has elapsed
                cv2.putText(img, '', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                message_displayed = False

    # Set the window property to stay in the foreground
    cv2.setWindowProperty('Webcam', cv2.WND_PROP_TOPMOST, cv2.WINDOW_NORMAL)

    cv2.imshow('Webcam', img)
    key = cv2.waitKey(5)
    if key == ord('q'):
        break

# Create "Attendance_File" folder if it doesn't exist
attendance_folder = os.path.join(os.getcwd(), 'Attendance_File')
if not os.path.exists(attendance_folder):
    os.makedirs(attendance_folder)

# Convert attendance.csv to Excel and save in the "Attendance_File" folder with subject_name
excel_filename = os.path.join(attendance_folder, f"{subject_name}_attendance.xlsx")
df = pd.read_csv("attendance.csv")
df.to_excel(excel_filename, index=False)
print(f"Attendance saved as {excel_filename}")

# Delete attendance.csv file after the program is executed
if os.path.exists("attendance.csv"):
    os.remove("attendance.csv")
    print("Deleted attendance.csv file.")

cv2.destroyAllWindows()
