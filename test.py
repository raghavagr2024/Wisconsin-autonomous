import cv2
import numpy as np

# Step 1: Load the image
image = cv2.imread('images/red.png')

# Step 2: Convert the image to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Step 3: Define the HSV range for red colors
lower_red1 = np.array([0, 100, 100])   # Lower range for red
upper_red1 = np.array([10, 255, 255])  # Upper range for light red
lower_red2 = np.array([160, 100, 100]) # Lower range for dark red
upper_red2 = np.array([180, 255, 255]) # Upper range for dark red

# Step 4: Create a mask for red colors
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(mask1, mask2)

# Step 5: Find contours from the red mask
contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 6: Loop over contours and filter for cone-like objects
cone_contours = []
for contour in contours:
    # Calculate the contour's area
    area = cv2.contourArea(contour)
    #ignore small objects
    if area < 350:
        continue
    
    # Approximate the contour to simplify it
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    
    # Check if the shape has at least 3 vertices (could be cone-like)
    if len(approx) >= 3:
        # Optionally, check for aspect ratio or area constraints
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        
        # Assuming cones are taller than they are wide
        if 0.3 < aspect_ratio < 1:  # Adjust ratio based on the cone's shape in your image
            cone_contours.append(contour)

# Step 7: Create an empty mask and draw only cone-like contours
cones_mask = np.zeros_like(image)

# Draw the cone contours on the mask
cv2.drawContours(cones_mask, cone_contours, -1, (255, 255, 255), thickness=cv2.FILLED)

# Step 8: Calculate centroids of the cone-like objects
centroids = []
for contour in cone_contours:
    # Calculate the moments
    M = cv2.moments(contour)
    if M["m00"] != 0:  # Avoid division by zero
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centroids.append((cX, cY))

#breaking into left and right zones
centroids_left = []
centroids_right = []

for centroid in centroids:
    if centroid[0] < image.shape[1] / 2:
        centroids_left.append(centroid)
    else:
        centroids_right.append(centroid)
print(centroids_left)
print(centroids_right)

#get best fit lines
[vxl, vyl, x0l, y0l] = cv2.fitLine(np.array(centroids_left), cv2.DIST_L2, 0, 0.01, 0.01)
[vxr, vyr, x0r, y0r] = cv2.fitLine(np.array(centroids_right), cv2.DIST_L2, 0, 0.01, 0.01)
point1l = (int(x0l - 1000 * vxl), int(y0l - 1000 * vyl))  # Start point of the line
point2l = (int(x0l + 1000 * vxl), int(y0l + 1000 * vyl))  # End point of the line
point1r = (int(x0r - 1000 * vxr), int(y0l - 1000 * vyr))  # Start point of the line
point2r = (int(x0r + 1000 * vxr), int(y0l + 1000 * vyr))  # End point of the line

# Step 9: Draw the lines on the image
cv2.line(image, point1l, point2l, (0, 0, 255), 2)
cv2.line(image, point1r, point2r, (0, 0, 255), 2)
# Step 10: Save the image
cv2.imwrite("images/answer.png", image)

