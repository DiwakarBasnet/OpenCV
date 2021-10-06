import cv2

# Faces and smile classifiers
face_detector = cv2.CascadeClassifier('Face_detector.xml')
smile_detector = cv2.CascadeClassifier('harcascade_smile.xml')

# Use webcam
webcam = cv2.VideoCapture(0)

# Show the current frame
while True:
  # Read current frame from webcam
  successful_frame_read, frame = webcam.read()

  # If there is an error, abort
  if not successful_frame_read:
    break

  # Change to grayscale
  frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Detect faces and smiles
  faces = face_detector.detectMultiScale(frame_grayscale)
  
  # Run smile detection in detected face
  for(x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4)
    the_face = frame[y:y+h, x:x+w]

    face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

    # scaleFactor --> blur black and white image because less detail means easy to detect smile
    # minNeisghbors --> has to be 20 neighbouring rectangles to be counted as a smile
    smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20)

    # Draw rectangle on face
    #for (x2, y2, w2, h2) in smiles:
    #  cv2.rectangle(the_face, (x2, y2), (x2 + w2, y2 + h2), (50, 50, 200), 3)

    # Label the face as smiling
    if len(smiles) > 0:
      cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))

  cv2.imshow('Are you smiling?', frame)

  cv2.waitKey(1)

# Cleanup
webcam.release()
cv2.destroyAllWindows()