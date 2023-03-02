import cv2
import numpy as np
import os
import time

# Part 1: Taking the video
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
cap.release()

# Part 3: Processing the frames
for frame_num in range(total_frames):
    # Read the image
    img = cv2.imread(f'Frame/frame_{frame_num}.jpg')

    # Step 1: Smoothing filter to reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Step 2: Thresholding filter to separate the background from the belt surface
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)

    # Step 3: Segmentation tool to segment the conveyor belt surface into positive and negative depths
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    if len(areas) > 1:
        max_idx = np.argmax(areas)
        cnt = contours[max_idx]
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [cnt], 0, 255, -1)
        positive_depth = cv2.bitwise_and(gray, gray, mask=mask)
        negative_depth = cv2.bitwise_and(gray, gray, mask=~mask)
    else:
        positive_depth = gray
        negative_depth = np.zeros_like(gray)

    # Step 4: Morphology filter to increase contrast between the two components
    kernel = np.ones((5,5), np.uint8)
    positive_depth = cv2.morphologyEx(positive_depth, cv2.MORPH_OPEN, kernel)
    negative_depth = cv2.morphologyEx(negative_depth, cv2.MORPH_OPEN, kernel)

    # Step 5: Edge detection filter to outline the positive and negative depths
    positive_depth_edges = cv2.Canny(positive_depth, 100, 200)
    negative_depth_edges = cv2.Canny(negative_depth, 100, 200)

    # Step 6: Pixel intensity tool to measure the depths
    positive_depth_intensity = np.mean(positive_depth[positive_depth_edges != 0])
    negative_depth_intensity = np.mean(negative_depth[negative_depth_edges != 0])

    # Step 7: Output the results
    print(f"Frame {frame_num}: Positive depth = {positive_depth_intensity:.2f}, Negative depth = {negative_depth_intensity:.2f}")


