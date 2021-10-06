import cv2

# Load the  face detecting algorithm
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect faces in
img = cv2.imread('avengers.jpg')
#img = cv2.imread('RDJ.jpg')
#img = cv2.imread('gipghy.gif')

# Convert image to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Draw rectangle around the faces
## Assigning coordinates to tuple
for (x, y, w, h) in face_coordinates:
  ### cv2.rectangle(img_name, (top_left_point_coordinates),   (width_and_height_of_square + previous_point = bot_right_point_coordinates), (color_of_box), thickness_of_rectangle)
  cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the image with rectangles on faces
cv2.imshow('Face Detector', img)

# Wait until key is pressed
cv2.waitKey(0)   ### k in key should be capital

print("Code completed")