import datetime
from flask import Flask, redirect, render_template, request, session, url_for, redirect
import cv2
import numpy as np
import os
import time
import sqlite3
import datetime

app = Flask(__name__, template_folder='template_folder')

def calc_cycle_length(speed, length):
    return length / speed * 5280 / 3600

@app.route('/', methods=['GET', 'POST'])
def index():
    speed = 0.0
    length = 0.0

    if request.method == 'POST':
        speed = float(request.form['speed'])
        length = float(request.form['length'])

    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # Get the values from the form
    speed = request.form.get('speed')
    length = request.form.get('length')
    
    # Check if both inputs are valid
    if speed is None or length is None:
        return "Invalid input"

    # Convert the inputs to floats
    speed = float(speed)
    length = float(length)

    # Calculate the cycle length
    cycle_length = calc_cycle_length(speed, length)
    print("The cycle length is", cycle_length, "seconds")

    # Create the folders
    for folder in ["Edge", "Frame", "Surface_defects", "Rips"]:
        os.makedirs(folder, exist_ok=True)

    # Delete the files in the folders
    for folder in ["Edge", "Frame", "Surface_defects", "Rips"]:
        for filename in os.listdir(folder):
            os.remove(os.path.join(folder, filename))

    # Delete the belt.mp4 file
    if os.path.exists('belt.mp4'):
        os.remove('belt.mp4')

    # Start recording the video
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('belt.mp4', fourcc, 20.0, (w,h))

    recording_time = cycle_length
    start_time = time.time()
    while(cap.isOpened() and time.time() - start_time < recording_time):
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Extract frames from the video
    cap = cv2.VideoCapture('belt.mp4')
    total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 300)

    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f'Frame/frame_{frame_num}.jpg', frame[:, 100:-100])
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
            sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=1)
            edges = cv2.Canny(image=img_blur, threshold1=25, threshold2=35)
        else:
            break

    # Get the list of all the images in the folder
    folder = 'Frame'
    images = [f for f in os.listdir(folder) if f.endswith('.jpg')]

    # Loop through all files in Frame folder
    for filename in images:
        # Load the image
        img = cv2.imread(os.path.join("Frame", filename))

        # Crop the left and right sides of the image
        cropped_img = img[:, 100:-100]

        # Overwrite the original image with the cropped image
        cv2.imwrite(os.path.join("Frame", filename), cropped_img)

        # Convert to grayscale and apply blur for better edge detection
        img_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0) 

        # Sobel Edge Detection on the Y axis
        sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=1)

        # Canny Edge Detection
        edges = cv2.Canny(image=img_blur, threshold1=25, threshold2=35)

    cv2.destroyAllWindows()


     # Part 2: saving edges to Edge folder

    # Make sure the Edge folder exists
    if not os.path.exists("Edge"):
        os.mkdir("Edge")

    # Loop through all files in Frame folder
    for filename in images:
        # Load the image
        img = cv2.imread(os.path.join("Frame", filename))

        # Convert to grayscale and apply blur for better edge detection
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

        # Canny Edge Detection
        edges = cv2.Canny(image=img_blur, threshold1=25, threshold2=35)

        # Save the edges image to the Edge folder
        cv2.imwrite(os.path.join("Edge", filename), edges)
        
    return redirect(url_for('results', cycle_length=cycle_length))

# Define total_frames
total_frames = 100

@app.route('/results', methods=['GET', 'POST'])
def results():
    # Set up a connection to a database
    conn = sqlite3.connect('results.db')
    c = conn.cursor()

    # Create a table to store the data
    c.execute('''CREATE TABLE IF NOT EXISTS results
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 avg_straightness REAL,
                 blue_values TEXT,
                 avg_num_vertices REAL,
                 avg_solidity REAL,
                 date TEXT,
                 time TEXT)''')

    # Get the current date and time
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.datetime.now().strftime('%H:%M:%S')

    # Part 3: testing straight line detection

    # Create an empty list to store all straightness values
    straightness_values = []
    surface_defects = [] # Initialize an empty list for surface defects

    # Loop through all files in Edge folder
    for filename in os.listdir("Edge"):
        # Load the image
        img_path = os.path.join("Edge", filename)
        if not os.path.isfile(img_path):
            print(f"Error: file {img_path} not found")
            continue
        
        img = cv2.imread(img_path)

        # Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Sobel Edge Detection on the Y axis
        sobely = cv2.Sobel(src=img_gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=1)

        # Calculate the straightness of the Sobel lines
        straightness = np.mean(np.abs(sobely))

        straightness_values.append(straightness)

    # Calculate the average of all straightness values
    avg_straightness = sum(straightness_values) / len(straightness_values) if len(straightness_values) > 0 else 0

    print(f"Final average straightness value: {avg_straightness:.2f}")

    # Part 3: In each frame, determine if there is any presence of rips
    if not os.path.exists('Rips'):
        os.makedirs('Rips')

    blue_values = []
    for frame_num in range(total_frames):
        # Read the image
        img_path = f'Frame/frame_{frame_num}.jpg'
        if not os.path.isfile(img_path):
            print(f"Error: file {img_path} not found")
            continue
        
        img = cv2.imread(img_path)

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

    # Part 5: Calculate the average overall surface condition of the belt
    num_good_frames = 0
    avg_num_vertices = 0
    avg_solidity = 0

    # Loop through all files in Surface folder
    for filename in os.listdir("Frame"):
        # Load the image
        img_path = os.path.join("Frame", filename)
        if not os.path.isfile(img_path):
            print(f"Error: file {img_path} not found")
            continue
    
    img = cv2.imread(img_path)

    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(img_gray, 100, 200)

    # Find contours in the image
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through all contours and extract relevant information
    for contour in contours:
        # Compute the number of vertices in the contour
        num_vertices = len(contour)

        # Compute the solidity of the contour
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
        else:
            solidity = 0

        # Add the surface defect to the list of surface defects
        surface_defects.append({'num_vertices': num_vertices, 'solidity': solidity})

    # Part 5: Calculate the average overall surface condition of the belt
    num_good_frames = 0
    avg_num_vertices = 0
    avg_solidity = 0
    for defect in surface_defects:
        if defect is not None:
            num_good_frames += 1
            avg_num_vertices += defect['num_vertices']
            avg_solidity += defect['solidity']
    avg_num_vertices /= num_good_frames if num_good_frames > 0 else 1
    avg_solidity /= num_good_frames if num_good_frames > 0 else 1

    print(f"Average overall surface condition of the belt: {avg_num_vertices:.2f} vertices, {avg_solidity:.2f} solidity")

       # Insert the data into the table
    c.execute('''INSERT INTO results
                (avg_straightness, blue_values, avg_num_vertices, avg_solidity, date, time)
                VALUES (?, ?, ?, ?, ?, ?)''',
                (avg_straightness, str(blue_values), avg_num_vertices, avg_solidity, current_date, current_time))

    # Save the changes to the database and close the connection
    conn.commit()
    conn.close()

    return render_template('results.html', avg_straightness=avg_straightness, blue_values=blue_values, avg_num_vertices=avg_num_vertices, avg_solidity=avg_solidity)

@app.route('/data')
def data():
    conn = sqlite3.connect('results.db')
    c = conn.cursor()

    # Retrieve all rows from the results table
    c.execute('SELECT * FROM results')
    rows = c.fetchall()

    # Close the connection
    conn.close()

    return render_template('data.html', rows=rows)