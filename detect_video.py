import cv2

# Create opencv video
#video = cv2.VideoCapture('Tesla Autopilot Dashcam.mp4')
video = cv2.VideoCapture('Pedestrians.mp4')

# Pre-trained classifier
car_tracker_file = 'cars.xml'
pedestrian_tracker_file = 'full_body.xml'

# create classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

# change video color to grayscale and run until the video ends
while True:
  # Read the current frame
  (read_successful, frame) = video.read()

  # Safe coding
  if read_successful:
    # Must convert to grayscale
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  else:
    break

  # Detect cars and pedestrians in current frame
  cars = car_tracker.detectMultiScale(grayscaled_frame)
  pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

  # Draw rectangles around the cars
  for (x, y, w, h) in cars:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
  # Draw rectangles around the pedestrians
  for (x, y, w, h) in pedestrians:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

  # Display the image with the cars and pedestrians spotted
  cv2.imshow('Car and Pedestrians detector', frame)

  # Dont autoclose (wait for key to be pressed)
  key = cv2.waitKey(1)

  # Stop if Q key is pressed
  if key == 81 or key == 113:
    break

# Release the VideoCapture object
video.release()
