import pandas as pd
import cv2
import urllib.request
import numpy as np
import os
from datetime import datetime
import face_recognition
import glob

path = r'C:\Users\anura\OneDrive\Desktop\attendance\attendance\image_folder'
url = 'http://192.168.135.194/cam-hi.jpg'

# List all Excel files in the "Attendance_File" folder
attendance_folder = os.path.join(os.getcwd(), 'Attendance_File')
excel_files = glob.glob(os.path.join(attendance_folder, '*.xlsx'))

# Show the available Excel files to the user
print("Available Excel Files:")
for i, file in enumerate(excel_files, start=1):
    print(f"{i}. {os.path.basename(file)}")

# Get user input for the selected Excel file
file_choice = int(input("Select the file to update (enter the corresponding number): ")) - 1
selected_excel_file = excel_files[file_choice]

# Get today's date in dd/mm/yy format
today = datetime.today().strftime('%d/%m/%y')

images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def extract_subject_name(file_name):
    # Assuming subject name is the starting word of the file name before an underscore
    return file_name.split('_')[0]

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name, period, date, excel_file):
    df = pd.read_excel(excel_file)  # Read the existing Excel file
    now = datetime.now()
    dtString = now.strftime('%H:%M:%S')
    
    # Check if the person is already marked present for today
    if not df[(df['Name'] == name) & (df['Date'] == date)].empty:
        print(f"{name} has already been marked present today.")
        return
    
    # Check if the person's name is already present in the Excel sheet
    if name in df['Name'].values:
        print(f"{name} is already present in the attendance sheet.")
        return
    
    # Add a new row for the attendance
    new_row = {'Subject': period, 'Name': name, 'Login Time': dtString, 'Date': date, 'Status': 'Present'}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Write the updated DataFrame back to the Excel file
    df.to_excel(excel_file, index=False)
    print(f"{name} marked present for {period} on {date}")


    
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Create the window with the flag cv2.WINDOW_NORMAL
cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Webcam', 800, 600)  # Set the size of the window


person_present = {}  # To keep track of persons and their box color
notification_printed = {}  # To keep track of whether notification has been printed

while True:
    try:
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

                # Check if the person is already marked present today
                if name not in notification_printed:
                    subject_name = extract_subject_name(os.path.splitext(os.path.basename(selected_excel_file))[0])

                    if name not in person_present:
                        markAttendance(name, subject_name, today, selected_excel_file)
                        person_present[name] = {'present': True}  # Mark the person as present
                    else:
                        person_present[name]['present'] = True

                    notification_printed[name] = True

                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                # Display a colored box based on whether the person is marked present or not
                color = (255, 0, 0) if person_present[name]['present'] else (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)


        # Update the status for persons not in the current frame
        for person in person_present.keys():
            if person not in [classNames[i].upper() for i in range(len(classNames))]:
                person_present[person]['present'] = False

        cv2.setWindowProperty('Webcam', cv2.WND_PROP_TOPMOST, cv2.WINDOW_NORMAL)
        cv2.imshow('Webcam', img)
        key = cv2.waitKey(5)
        if key == ord('q'):
            break

    except Exception as e:
        print(f"Error: {e}")

cv2.destroyAllWindows()
