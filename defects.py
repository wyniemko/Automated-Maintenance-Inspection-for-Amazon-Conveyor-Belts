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

# Part 4: Defect Analysis
def detect_defects(frame, image_name):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Material defect detection
    thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    material_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            material_count += 1
            cv2.drawContours(frame, [contour], 0, (0, 0, 255), 2)

    # Rip defect detection
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    rip_count = 0
    for x in range(sobelx.shape[1]):
        col = sobelx[:, x]
        for y in range(col.shape[0] - 1):
            if col[y] < 0 and col[y + 1] >= 0:
                rip_count += 1
                cv2.circle(frame, (x, y), 5, (0, 255, 0), 2)
                # Save rip defect screenshot to separate folder
                output_path = os.path.join('Rip_defects', image_name.split('.')[0] + f'_rip_{rip_count}.jpg')
                cv2.imwrite(output_path, frame[y-50:y+50, x-50:x+50])

    # Amnesty defect detection
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    amnesty_count = 0
    for y in range(sobely.shape[0]):
        row = sobely[y, :]
        for x in range(row.shape[0] - 1):
            if row[x] > 0 and row[x + 1] <= 0:
                amnesty_count += 1
                cv2.circle(frame, (x, y), 5, (255, 0, 0), 2)
                # Save amnesty defect screenshot to separate folder
                output_path = os.path.join('Amnesty_defects', image_name.split('.')[0] + f'_amnesty_{amnesty_count}.jpg')
                cv2.imwrite(output_path, frame[y-50:y+50, x-50:x+50])

    # Return frame and defect counts
    return frame, material_count, rip_count, amnesty_count

# Create output folders for defect screenshots
os.makedirs('Material_defects', exist_ok=True)
os.makedirs('Rip_defects', exist_ok=True)
os.makedirs('Amnesty_defects', exist_ok=True)

# Get the list of all the images in the folder
folder = 'Defect_frames'
output_folder = 'Output'
os.makedirs(output_folder, exist_ok=True)
images = [f for f in os.listdir(folder) if f.endswith('.jpg')]

for image_name in images:
    # Load the image
    frame = cv2.imread(os.path.join(folder, image_name))

# Detect the defects
output, material_count, rip_count, amnesty_count = detect_defects(frame)

# Save the output image
output_path = os.path.join(output_folder, image_name)
cv2.imwrite(output_path, output)

# Print the defect counts
print(f'{image_name}: Material = {material_count}, Rip = {rip_count}, Amnesty = {amnesty_count}')

# Create defect output folders
material_folder = os.path.join(output_folder, 'material_defects')
rip_folder = os.path.join(output_folder, 'rip_defects')
amnesty_folder = os.path.join(output_folder, 'amnesty_defects')
os.makedirs(material_folder, exist_ok=True)
os.makedirs(rip_folder, exist_ok=True)
os.makedirs(amnesty_folder, exist_ok=True)

# Save defect screenshots to respective folders
for i, count in enumerate([material_count, rip_count, amnesty_count]):
    if count > 0:
        defect_folder = [material_folder, rip_folder, amnesty_folder][i]
        defect_name = image_name.split('.')[0] + f'_defect_{i}.jpg'
        defect_path = os.path.join(defect_folder, defect_name)
        cv2.imwrite(defect_path, output)

