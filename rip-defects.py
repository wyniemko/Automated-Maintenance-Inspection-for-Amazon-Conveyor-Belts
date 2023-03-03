import cv2
import numpy as np
import os
import time

#create two empty folders called Edge and Frame
if not os.path.exists('Rips'):
    os.makedirs('Rips')
if not os.path.exists('Frame'):
    os.makedirs('Frame')

#delete the files in the folders called Edge and Frame
for filename in os.listdir("Rips"):
    os.remove(os.path.join("Rips", filename))
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

# Part 3: In each frame, determine if there is any presence of rips

if not os.path.exists('Rips'):
    os.makedirs('Rips')

blue_values = []
for frame_num in range(total_frames):
    # Read the image
    img = cv2.imread(f'Frame/frame_{frame_num}.jpg')

# Convert the image to HSV color space
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the range of blue color in HSV
lower_blue = np.array([100,50,50])
upper_blue = np.array([130,255,255])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

# Count the number of blue pixels in the image
blue_pixels = np.sum(mask == 255)
blue_values.append(blue_pixels)

# Save the image with blue pixels as a screenshot
if blue_pixels > 0:
    cv2.imwrite(f'Rips/rip_{frame_num}.jpg', img)

# Part 4: List all of the Rips, from largest to smallest.

# Get the indices of the blue values in descending order
sorted_indices = np.argsort(blue_values)[::-1]

print("Rips sorted by size:")
for index in sorted_indices:
    if blue_values[index] > 0:
        print(f"rip_{index}.jpg: {blue_values[index]} blue pixels")
