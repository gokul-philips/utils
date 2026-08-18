import cv2
import os
import numpy as np
from screeninfo import get_monitors
import time
import platform

# Define paths
path1 = r"C:\Users\320329176\Downloads\MR_Clean\WorkflowCamera\unextracted_out"
path2 = r"C:\Users\320329176\Downloads\MR_Clean\WorkflowCamera\WorkflowCameraOut\patientImage"

# Ensure the script runs on Windows
if platform.system() != 'Windows':
    raise Exception("This script is designed to run on Windows only.")

# List files in both directories
fns1 = os.listdir(path1)
fns2 = os.listdir(path2)
print(f"Files in path1: {fns1}")
print(f"Files in path2: {fns2}")

fns = [fn for fn in fns1 if fn in fns2]
print(f"Common files: {fns}")

# Check if there are common files
if not fns:
    raise Exception("No common files found in the provided directories.")

# Get monitor information
monitors = get_monitors()
print("Available monitors:", monitors)
if len(monitors) < 2:
     # Use the second monitor
    primary_monitor = monitors[0]
    monitor_x = primary_monitor.x
    monitor_y = primary_monitor.y
    monitor_width = primary_monitor.width
    monitor_height = primary_monitor.height   
else:
    # Use the second monitor
    second_monitor = monitors[1]
    monitor_x = second_monitor.x
    monitor_y = second_monitor.y
    monitor_width = second_monitor.width
    monitor_height = second_monitor.height

# Create a named window
cv2.namedWindow("Sample", cv2.WINDOW_NORMAL)

# Move the window to the second monitor
cv2.moveWindow("Sample", monitor_x, monitor_y)

# Resize the window to fit your second monitor's resolution
cv2.resizeWindow("Sample", monitor_width, monitor_height)

# Wait for a short period to ensure the window is displayed correctly
time.sleep(1)

# Resize the window again to make sure it fills the screen
cv2.resizeWindow("Sample", monitor_width, monitor_height)

# Function to resize images while maintaining aspect ratio
def resize_image(image, max_width, max_height):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)
    return cv2.resize(image, (int(w * scale), int(h * scale)))



# Loop through the images and display them side by side
i = 0
while True:
    fn = fns[i]
    fn1 = os.path.join(path1, fn)
    fn2 = os.path.join(path2, fn)
    # fn1 = fn1.strip()
    # fn2 = fn2.strip()
    # fn1 = fn1.encode('utf-8').decode('utf-8')
    # fn2 = fn2.encode('utf-8').decode('utf-8')
    # fn1 = fn1.replace('\\', '/')
    # fn2 = fn2.replace('\\', '/')
    # print(os.path.abspath(fn1))
    # print(repr(fn1))
    # print(repr(fn2))

    # from pathlib import Path
    # fn1_path = Path(fn1)
    # fn2_path = Path(fn2)
    # if fn1_path.exists() and fn2_path.exists():
    #     print("Files exist")


    if os.path.exists(fn1) and os.path.exists(fn2):
        img1 = cv2.imread(fn1, -1)
        img2 = cv2.imread(fn2, -1)

        if img1 is None or img2 is None:
            print(f"Error loading images: {fn1}, {fn2}")
            i = (i + 1) % len(fns)
            continue

        img1_resized = resize_image(img1, monitor_width // 2, monitor_height)
        img2_resized = resize_image(img2, monitor_width // 2, monitor_height)

        canvas = np.zeros((monitor_height, monitor_width, 3), dtype=np.uint8)
        canvas[:img1_resized.shape[0], :img1_resized.shape[1]] = img1_resized
        canvas[:img2_resized.shape[0], monitor_width // 2:monitor_width // 2 + img2_resized.shape[1]] = img2_resized

        cv2.imshow("Sample", canvas)

        k = cv2.waitKey(1) & 0xFF  # Short delay and mask to handle key press detection

        if k == ord('x') or k == ord('q'):
            break
        elif k == ord('n'):
            i = (i + 1) % len(fns)
        elif k == ord('p'):
            i = (i - 1) % len(fns)

cv2.destroyAllWindows()

