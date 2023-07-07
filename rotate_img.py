import cv2

# Load the image
image = cv2.imread("dataset_test/precanny4.png")

# Specify the rotation angle in degrees
angle = 271

# Get the image dimensions
height, width = image.shape[:2]

# Calculate the rotation matrix
rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)

# Apply the rotation to the image
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

# Display the original and rotated images
cv2.imshow("Original Image", image)
cv2.imshow("Rotated Image", rotated_image)
# cv2.imwrite("plane_vert.png", rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()