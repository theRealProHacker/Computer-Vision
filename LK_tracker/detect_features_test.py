import cv2
import numpy as np

# Open the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Parameters for corner detection
feature_params = dict(
    maxCorners=200,
    qualityLevel=0.01,
    minDistance=3,
    blockSize=5
)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture an image.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Preprocess the image
    gray = cv2.equalizeHist(gray)  # Equalize histogram for better contrast
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Smooth the image

    # Detect corners
    corners = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)

    if corners is not None:
        # Draw detected corners on the frame
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    # Show the frame
    cv2.imshow("Corners", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
