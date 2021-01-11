import cv2

#Load some pre-trained data on face frontals from opencv(haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Capture video from webcam
webcam = cv2.VideoCapture(0)

#Iterate forever over the frames
while True:

    #Read the current frame
    successful_frame_read, frame = webcam.read()

    #Convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect objects of different sizes in the input image
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #Draw rectangles aorund the faces. The last numbres is the thicknes of the line
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
  
    cv2.imshow('Face_detector_app', frame)
    key = cv2.waitKey(1)

    #Stop is Q is pressed
    if key==81 or key==113:
        break

#Release the videocaptured object
webcam.release()
