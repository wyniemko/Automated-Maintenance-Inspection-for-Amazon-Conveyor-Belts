import cv2
import numpy as np
import os
import time

#create two empty folders called Edge and Frame
if not os.path.exists('Suface_defects'):
    os.makedirs('Suface_defects')
if not os.path.exists('Frame'):
    os.makedirs('Frame')

#delete the files in the folders called Edge and Frame
for filename in os.listdir("Suface_defects"):
    os.remove(os.path.join("Suface_defects", filename))
for filename in os.listdir("Frame"):
    os.remove(os.path.join("Frame", filename))

#Part 1: Taking the video
def calc_cycle_length(speed, length):
    return length/speed * 5280 / 3600

if __name__ == "__main__":
    speed = float(input("Enter the speed of the belt in mph: "))
    length = float(input("Enter the length of the belt in feet: "))

    cycle_length = calc_cycle_length(speed, length)
    print("The cycle length is", cycle_length, "seconds")
    
    cap = cv2.VideoCapture(0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('belt.mp4',fourcc, 20.0, (640,480))
    
    recording_time = cycle_length
    start_time = time.time()
    while(cap.isOpened() and time.time() - start_time < recording_time):
        ret, frame = cap.read()
        if ret==True:
            out.write(frame)

            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Part 2: Extracting frames from the video

# Load the video file
cap = cv2.VideoCapture('belt.mp4')

# Get the number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Limit the number of frames to 300
total_frames = min(total_frames, 300)

# Create a folder to store the extracted frames
if not os.path.exists('Frame'):
    os.makedirs('Frame')

# Extract each frame and save it as an image
for frame_num in range(total_frames):
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(f'Frame/frame_{frame_num}.jpg', frame)
    else:
        break

# Release the VideoCapture object
cap.release()

# Part 3: In each frame, determine the surface conditon of the belt.

def detect_surface_defect(frame):
# Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection to detect edges
    edges = cv2.Canny(blur, 50, 150)

# Apply thresholding to separate the belt from the background
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Check if there are any contours
    if contours:
    # Find the contour with the largest area
        c = max(contours, key=cv2.contourArea)

    # Calculate the perimeter of the contour
    perimeter = cv2.arcLength(c, True)

    # Approximate the contour to a polygon
    approx = cv2.approxPolyDP(c, 0.04 * perimeter, True)

    # Calculate the number of vertices of the polygon
    num_vertices = len(approx)

    # Check if the polygon is convex
    is_convex = cv2.isContourConvex(approx)

    # Calculate the solidity of the contour
    area = cv2.contourArea(c)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area

    # Return a dictionary with the surface defect information
    return {
        'edges': edges,
        'num_vertices': num_vertices,
        'is_convex': is_convex,
        'solidity': solidity
    }

    # If there are no contours, return None
    return None

# Part 4: Load the frames and detect the surface defects

frames = []
surface_defects = []
for frame_num in range(total_frames):
    frame = cv2.imread(f'Frame/frame_{frame_num}.jpg')
frames.append(frame)

# Detect the surface defects in the frame
surface_defect = detect_surface_defect(frame)
surface_defects.append(surface_defect)

# Part 5: Calculate the average overall surface condition of the belt and determine if the belt is in good condition or not.

num_edges = []
num_vertices = []
is_convex = []
solidity = []

for surface_defect in surface_defects:
    if surface_defect is not None:
        num_edges.append(np.sum(surface_defect['edges']) / 255)
num_vertices.append(surface_defect['num_vertices'])
is_convex.append(surface_defect['is_convex'])
solidity.append(surface_defect['solidity'])

# Calculate the average values

avg_num_edges = np.mean(num_edges)
avg_num_vertices = np.mean(num_vertices)
avg_is_convex = np.mean(is_convex)
avg_solidity = np.mean(solidity)

# Determine if the belt is in good condition or not

if avg_num_edges < 50 and avg_num_vertices >= 4 and avg_is_convex and avg_solidity > 0.9:
    print("The belt is in good condition")
else:
    print("The belt is in poor condition")
