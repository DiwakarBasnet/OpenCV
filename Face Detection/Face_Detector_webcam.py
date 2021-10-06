import cv2

# Load pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Capture video from webcam
webcam = cv2.VideoCapture(0)

### Iterate forever over frames
while True:

  ### Read the current frame
  (successful_frame_read, frame) = webcam.read()

  # Safe coding
  if successful_frame_read:
    # Convert to grayscale
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  else:
    break

  # Detect faces
  face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)

  # Draw rectangles around the faces
  for (x, y, w, h) in face_coordinates:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # cv2.rectangle(img_name, (top_left_point_coordinates), (width_and_height_of_square + previous_point = bot_right_point_coordinates), (color_of_box), thickness_of_rectangle)
  
    # Display the video of faces getting detected
    cv2.imshow('Face Detector', frame)
    # Wait until key is pressed, by putting 1 it automatically hits a key in 1 ms
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

### Release the VideoCapture object
webcam.release()

# De-allocate any associated memory usage 
cv2.destroyAllWindows()  