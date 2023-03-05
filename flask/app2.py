from flask import Flask, redirect, render_template, request, session, url_for, redirect
import cv2
import numpy as np
import os
import time

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
            cv2.imwrite(os.path.join("Edge", f"Edge Detection_{frame_num}.jpg"), edges)
        else:
            break
        
    return redirect(url_for('results', cycle_length=cycle_length))

@app.route('/results', methods=['GET', 'POST'])
def results():
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

    # Part 3: testing straight line detection

    # Create an empty list to store all straightness values
    straightness_values = []

    # Loop through all files in Edge folder
    for filename in os.listdir("Edge"):
        # Load the image
        img = cv2.imread(os.path.join("Edge", filename))

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

    return render_template('results.html', avg_straightness=avg_straightness)
