from flask import Flask, render_template, request, send_from_directory
import cv2
import numpy as np
import os
import time
import json

app = Flask(__name__)

@app.route('/run_script', methods=['POST', 'GET'])
def run_script():
    if request.method == 'POST':
        # Get input values from form
        speed = float(request.form['speed'])
        length = float(request.form['length'])
        
        # Call your script here with the input values
        result = {
            'Edge': 'C:\temp\flask-app\Edge',
            'Frame': 'C:\temp\flask-app\Frame',
            'Rips': 'C:\temp\flask-app\Rips',
            'Surface_defects': 'path/to/Surface_defects/image'
        }
        return json.dumps(result)
    else:
        # Display the form for user input
        return '''
            <form method="post">
                <p>Enter the speed of the belt in mph: <input type="text" name="speed"></p>
                <p>Enter the length of the belt in feet: <input type="text" name="length"></p>
                <p><input type="submit" value="Submit"></p>
            </form>
        '''

#create two empty folders called Edge and Frame
if not os.path.exists('Edge'):
    os.makedirs('Edge')
if not os.path.exists('Frame'):
    os.makedirs('Frame')
if not os.path.exists('Suface_defects'):
    os.makedirs('Suface_defects')
if not os.path.exists('Rips'):
    os.makedirs('Rips')

#delete the files in the folders called Edge and Frame
for filename in os.listdir("Edge"):
    os.remove(os.path.join("Edge", filename))
for filename in os.listdir("Frame"):
    os.remove(os.path.join("Frame", filename))
for filename in os.listdir("Suface_defects"):
    os.remove(os.path.join("Suface_defects", filename))
for filename in os.listdir("Rips"):
    os.remove(os.path.join("Rips", filename))

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

# Get the list of all the images in the folder
folder = 'Frame'
images = [f for f in os.listdir(folder) if f.endswith('.jpg')]

# Loop through all files in Frame folder
for filename in os.listdir("Frame"):
    # Load the image
    img = cv2.imread(os.path.join("Frame", filename))

    # Crop the left and right sides of the image
    cropped_img = img[:, 100:-100]

    # Overwrite the original image with the cropped image
    cv2.imwrite(os.path.join("Frame", filename), cropped_img)

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

# Part 3: testing straight line detection

# Create an empty list to store all straightness values
straightness_values = []

# Loop through all files in Edge folder
for filename in os.listdir("Edge"):
    # Load the image
    img = cv2.imread(os.path.join("Edge", filename))

    # Crop the left and right sides of the image
    cropped_img = img[:, 100:-100]

    # Overwrite the original image with the cropped image
    cv2.imwrite(os.path.join("Edge", filename), cropped_img)

    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sobel Edge Detection on the Y axis
    sobely = cv2.Sobel(src=img_gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=1)

    # Calculate the straightness of the Sobel lines
    straightness = np.mean(np.abs(sobely))

    straightness_values.append(straightness)

# Calculate the average of all straightness values
avg_straightness = sum(straightness_values) / len(straightness_values) if len(straightness_values) > 0 else 0

print("\n\n" +
      f"Final average straightness value: {avg_straightness:.2f}\n" +
      ("The straightness belt is on track." if avg_straightness >= 0.75 else
       "The conveyor belt is slightly misaligned. Tighten the belt." if avg_straightness >= 0.50 else
       "The conveyor belt is misaligned. Replace the belt & add more tension.") +
      "\n\n")

# Part 4: In each frame, determine the surface conditon of the belt.

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

# Part 5: Load the frames and detect the surface defects

frames = []
surface_defects = []
for frame_num in range(total_frames):
    frame = cv2.imread(f'Frame/frame_{frame_num}.jpg')
    frames.append(frame)

    # Detect the surface defects in the frame
    surface_defect = detect_surface_defect(frame)
    surface_defects.append(surface_defect)


# Detect the surface defects in the frame
surface_defect = detect_surface_defect(frame)
surface_defects.append(surface_defect)

# Part 6: Calculate the average overall surface condition of the belt and determine if the belt is in good condition or not.

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

# Calculate the average values
avg_num_edges = np.mean(num_edges)
avg_num_vertices = np.mean(num_vertices)
avg_is_convex = np.mean(is_convex)
avg_solidity = np.mean(solidity)

# Part 7: In each frame, determine if there is any presence of rips

blue_values = []
for frame_num in range(total_frames):
    # Read the image
    img = cv2.imread(f'Frame/frame_{frame_num}.jpg')

    # Convert the image to HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the range of blue color in HSV
    lower_blue = np.array([100,100,100])
    upper_blue = np.array([130,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

    # Count the number of blue pixels in the image
    blue_pixels = np.sum(mask == 255)
    blue_values.append(blue_pixels)

    # Save the image with blue pixels as a screenshot
    if blue_pixels > 0:
        cv2.imwrite(f'Rips/rip_{frame_num}.jpg', img)

# Part 8: List all of the Rips, from largest to smallest.

# Get the indices of the blue values in descending order
sorted_indices = np.argsort(blue_values)[::-1]

print('====================================================================================================')
print('=                                    RIPS STORTED BY SIZE                                            =')
print('====================================================================================================\n')


print("Rips sorted by size:")
for index in sorted_indices:
    if blue_values[index] > 0:
        print(f"rip_{index}.jpg: {blue_values[index]} blue pixels")
    else:
        print("No rips detected")

# Output the average values in a box
print('\n===================================================================================================')
print('=                           DESCRIPTION AND USE IN BELT CONDITION EVALUATION                          =')
print('====================================================================================================\n')

print("Edge: The number of edges detected on the belt surface. A lower number of edges is indicative of a smoother surface, while a higher number of edges may indicate damage or debris on the belt.\n")

print("Vertices: The number of vertices detected on the belt surface. A higher number of vertices is indicative of a rougher surface, which can cause wear and tear on the belt over time.\n")

print("Convexity: A measure of how 'bulging' or 'hollow' the belt surface is. A convex surface is one that curves outward, while a concave surface curves inward. In general, a more convex surface is less likely to cause damage to the belt.\n")

print("Solidity: The ratio of the area of the belt surface to the area of its convex hull. This measures how 'solid' the belt surface is, with higher values indicating a smoother, less irregular surface.\n")

print('====================================================================================================')
print('=                        AVERAGE EVALUATIONS OF BELT SURFACE CONDITION                                =')
print('====================================================================================================\n')

print(f"Average number of edges: {avg_num_edges:.2f} (Minimum accepted value: 50)\n")
print(f"Average number of vertices: {avg_num_vertices:.2f}\n (Minimum accepted value: 0.9)\n")
print(f"Average convexity: {avg_is_convex:.2f}\n (Minimum accepted value: 0.9)\n") 
print(f"Average solidity: {avg_solidity:.2f} (Minimum accepted value: 0.9)\n")

if avg_num_edges < 50 and avg_num_vertices >= 4 and avg_is_convex and avg_solidity > 0.9:
    print("The belt surface is in good condition!")
else:
    print("The belt surface is in poor condition. Clean the belt and try again.")


print('====================================================================================================')
print('=                                              BELT STRAIGHTNESS                                      =')
print('====================================================================================================\n')
 
