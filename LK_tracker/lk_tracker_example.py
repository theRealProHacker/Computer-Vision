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
    minDistance=7,
    blockSize=7
)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Create random colors for tracks
colors = np.random.randint(0, 255, (1000, 3))

# Initialize variables
old_gray = None
p0 = None
mask = None

print("Press 'q' to quit, 'n' to reinitialize points.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture an image.")
        break

    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if old_gray is None:
        # First frame: initialize tracking
        old_gray = frame_gray
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        mask = np.zeros_like(frame)

    if p0 is not None and len(p0) > 0:
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        if p1 is not None and st is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                color = colors[i % len(colors)].tolist()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color, 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, color, -1)

            # Update the previous points and frame
            p0 = good_new.reshape(-1, 1, 2)
            old_gray = frame_gray.copy()
        else:
            # No good points to track
            p0 = None
    else:
        # If no points are left, keep displaying the feed
        pass

    # Overlay the mask on the frame
    output = cv2.add(frame, mask)

    # Display the frame
    cv2.imshow("Camera Feed with Lucas-Kanade Tracker", output)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('n'):  # Reinitialize points
        print("Reinitializing points...")
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        old_gray = frame_gray.copy()
        mask = np.zeros_like(frame)  # Reset the mask for new points

cap.release()
cv2.destroyAllWindows()
