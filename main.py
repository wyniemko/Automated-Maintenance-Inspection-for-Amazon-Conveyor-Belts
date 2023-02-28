import cv2
import numpy as np
import os
import time

#create two empty folders called Edge and Frame
if not os.path.exists('Edge'):
    os.makedirs('Edge')
if not os.path.exists('Frame'):
    os.makedirs('Frame')

#delete the files in the folders called Edge and Frame
for filename in os.listdir("Edge"):
    os.remove(os.path.join("Edge", filename))
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

# Part 3: Masking the image
mask_size = 300

# Get the list of all the images in the folder
folder = 'Frame'
images = [f for f in os.listdir(folder) if f.endswith('.jpg')]

for image_name in images:
    # Load the image
    frame = cv2.imread(os.path.join(folder, image_name))

    # Get the height and width of the image
    height, width = frame.shape[:2]

    # Calculate the top-left and bottom-right coordinates of the mask
    x1 = (width - mask_size) // 2
    y1 = (height - mask_size) // 2
    x2 = x1 + mask_size
    y2 = y1 + mask_size

    # Create a black image with the same size as the frame
    mask = np.zeros((height, width, 3), np.uint8)

    # Draw a white rectangle on the mask to cover everything outside of the middle square
    cv2.rectangle(mask, (0, 0), (width, y1), (255, 255, 255), -1)
    cv2.rectangle(mask, (0, y2), (width, height), (255, 255, 255), -1)
    cv2.rectangle(mask, (0, y1), (x1, y2), (255, 255, 255), -1)
    cv2.rectangle(mask, (x2, y1), (width, y2), (255, 255, 255), -1)

    # Invert the mask to make it clear
    mask = cv2.bitwise_not(mask)

    # Apply the mask to the frame
    result = cv2.bitwise_and(frame, mask)

    # Save the masked image
    cv2.imwrite(os.path.join(folder, image_name), result)

# part 4: Edge detection for all the images in Frame folder

# Create a folder to store the edge-detected frames
if not os.path.exists('Edge'):
    os.makedirs('Edge')

# Loop through all files in Frame folder
for filename in os.listdir("Frame"):
    # Load the image
    img = cv2.imread(os.path.join("Frame", filename))

    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0) 
    
    # Sobel Edge Detection on the Y axis
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=1)
    
    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=25, threshold2=35)
    
    # Save the image with the edge detection result
    cv2.imwrite(os.path.join("Edge", f"Edge Detection_{filename}"), edges)
 
cv2.destroyAllWindows()


# Part 5: testing straight line detection

# Create an empty list to store all straightness values
straightness_values = []

# Loop through all files in Edge folder
for filename in os.listdir("Edge"):
    # Load the image
    img = cv2.imread(os.path.join("Edge", filename))

    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect vertical lines using Hough transform
    lines = cv2.HoughLinesP(img_gray, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    # Calculate the average distance between endpoints of detected lines
    if lines is not None:
        endpoint_dists = [abs(line[0][2]-line[0][0]) for line in lines if abs(line[0][3]-line[0][1]) > 20]
        avg_dist = sum(endpoint_dists) / len(endpoint_dists) if len(endpoint_dists) > 0 else 0

        # Calculate the straightness of the vertical edge
        straightness = 1 / avg_dist if avg_dist > 0 else 0

        straightness_values.append(straightness)

        print(f"Straightness of {filename}: {straightness:.2f}")
    else:
        print(f"No vertical lines detected in {filename}, move camera further left.")

# Calculate the average of all straightness values
avg_straightness = sum(straightness_values) / len(straightness_values) if len(straightness_values) > 0 else 0

print("\n\n" +
      f"Final average straightness value: {avg_straightness:.2f}\n" +
      ("The straightness belt is on track." if avg_straightness >= 0.75 else
       "The conveyor belt is slightly misaligned. Tighten the belt." if avg_straightness >= 0.50 else
       "The conveyor belt is misaligned. Replace the belt & add more tension.") +
      "\n\n")
