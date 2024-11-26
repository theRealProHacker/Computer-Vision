import cv2

# Open a connection to the default camera (usually the built-in webcam)
cap = cv2.VideoCapture(0)  # 0 refers to the default camera; use 1 or higher for external cameras

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

print("Press 'q' to quit.")

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Check if the frame was captured successfully
    if not ret:
        print("Error: Failed to capture an image.")
        break

    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the frame in a window
    cv2.imshow('Camera Feed', gray)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
