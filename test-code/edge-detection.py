import cv2
import numpy as np
import os
import time

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

# Part 3: Masking the image

# Define the size of the mask
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

# Part 4: Detecting the vertical length of the gray belt

# Create a list to store the sum of white pixels in each contour
pixel_sums = []

# Loop over all images in the folder "Frame"
frame_counter = 0
for filename in os.listdir("Frame"):
    # Load the image
    img = cv2.imread(os.path.join("Frame", filename))

    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
    
    # Sobel Edge Detection
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=1) # Sobel Edge Detection on the Y axis
    # Display Sobel Edge Detection Images
    cv2.waitKey(0)
    
    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=25, threshold2=35) # Canny Edge Detection
    # Display Canny Edge Detection Image
    
    cv2.imshow('Canny Edge Detection', edges)
    cv2.waitKey(0)
 
cv2.destroyAllWindows()

# Part 5: Calculating the pixels of the lines

# Create a list to store the sum of white pixels in each contour
pixel_sums = []

# Loop over all images in the folder "Frame"
frame_counter = 0
for filename in os.listdir("Frame"):
    # Load the image
    img = cv2.imread(os.path.join("Frame", filename))

    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
    
    # Sobel Edge Detection
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=1) # Sobel Edge Detection on the Y axis
    # Display Sobel Edge Detection Images
    cv2.waitKey(0)
    
    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=25, threshold2=35) # Canny Edge Detection
    
    # Display Canny Edge Detection Image
    cv2.imshow('Canny Edge Detection', edges)
    cv2.waitKey(0)

    # Find the contours in the image
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create a black image with the same size as the frame
    mask = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    # Loop over all contours
    for contour in contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Check if the contour is a line
        if w > 10 and h < 10:
            # Draw the contour on the mask
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
    
    # Convert the mask to grayscale
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # Calculate the sum of white pixels in the mask
    pixel_sum = np.sum(mask_gray == 255)
    # Add the sum to the list
    pixel_sums.append(pixel_sum)

    # Save the masked image
    cv2.imwrite(os.path.join("Frame", filename), mask)

    # Display the masked image
    cv2.imshow('Mask', mask)
    cv2.waitKey(0)

    # Increment the frame counter  
    frame_counter += 1


# Close all windows
cv2.destroyAllWindows()
