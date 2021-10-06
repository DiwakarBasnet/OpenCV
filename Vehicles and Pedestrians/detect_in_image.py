import cv2

# create opencv image
img = cv2.imread('highway.png')
#img = cv2.imread('pedestrian.jpg')

# Pre-trained classifier
car_tracker_file = 'cars.xml'
pedestrian_tracker_file = 'full_body.xml'

# create classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

# Convert to grayscale
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detect cars and pedestrians
cars = car_tracker.detectMultiScale(black_n_white)
pedestrians = pedestrian_tracker.detectMultiScale(black_n_white)

# Draw rectangles around the cars
for (x, y, w, h) in cars:
  cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
  cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
# Draw rectangles around the pedestrians
for (x, y, w, h) in pedestrians:
  cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)

# Display the image with the cars and pedestrians spotted
cv2.imshow('Car and Pedestrians detector', img)

# Image is shown until a key is pressed
cv2.waitKey()