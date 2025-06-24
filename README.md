# Attendance-System-IOT
This is a Python-based automated attendance system that uses face recognition technology to mark student or employee attendance. The system captures a live video stream from a ESP 32 Cam Module (via IP webcam) it detects faces, recognizes registered individuals, and records their attendance in an Excel file. 

Features:
Live video feed in laptop via Esp32 cam module, 
Face detection and recognition using face_recognition and OpenCV,
Automatically marks attendance in a selected Excel sheet,
Prevents duplicate entries for the same day,
Displays real-time feedback on webcam view (e.g., “Marked present for Anurag”),
Organizes attendance files in a dedicated folder

Technologies Used
Python
OpenCV
face_recognition
Pandas
ESP32 Cam 

This system  can then be used to track attendance for a variety of  purposes, such as tracking attendance for school, work,  or events.
