# Amazon Conveyor Belt Tracking

This system is designed to monitor an Amazon conveyor belt, by capturing, processing, and analyzing its complete length, in order to identify any anomalies in the straightness of the belt edges. The process begins by extracting frames from the video footage, then applying a masking technique to isolate the relevant images. The edges of the images are then detected and assessed to determine the level of straightness. This information is subsequently utilized to generate a belt health metric, which provides a quantitative assessment of the conveyor belt's performance. Based on the analysis, the system offers recommendations for appropriate corrective actions, should they be necessary.

## Requirements

- Python 3
- OpenCV
- Numpy

## Utilization of OpenCV Functions

- cv2.VideoCapture(): to access the video capture device to record the video.
- cv2.VideoWriter_fourcc(): to select the video codec for the video writer object.
- cv2.VideoWriter(): to create a video writer object and save the video.
- cv2.imread(): to load an image from a file.
- cv2.imwrite(): to write an image to a file.
- cv2.cvtColor(): to convert the color space of an image from one color space to another.
- cv2.GaussianBlur(): to apply a Gaussian blur filter to an image.
- cv2.Sobel(): to perform edge detection on an image using the Sobel operator.
- cv2.Canny(): to perform edge detection on an image using the Canny operator.
- cv2.HoughLinesP(): to detect straight lines in an image using the probabilistic Hough transform.
- cv2.rectangle(): to draw a rectangle on an image.
- cv2.bitwise_not(): to invert the pixels of an image.
- cv2.bitwise_and(): to perform bitwise logical "AND" operation between the pixels of two images.
- np.zeros(): to create an array of zeros with a specified shape.
- np.uint8(): to convert an array of numbers to 8-bit unsigned integers.
- np.pi: a constant that represents the value of pi.
- os.listdir(): to get the list of files in a directory.
- os.path.join(): to join two or more path components.

## Part 1: Video Capture

In the first part of the code, the user is prompted to enter the speed and length of a conveyor belt. The cycle length of the belt is then calculated using the formula:

cycle_length = length / speed * 5280 / 3600

The video codec used to compress the video file is set to mp4v using the cv2.VideoWriter_fourcc function. The total number of frames in the video is obtained using the cv2.VideoCapture.get function with the cv2.CAP_PROP_FRAME_COUNT argument. The images are read and written to files using the cv2.imread and cv2.imwrite functions, respectively. The recording stops after the calculated cycle length has elapsed.

## Part 2: Frame Extraction

In this part, the video file is loaded using `cv2.VideoCapture` and the number of frames in the video is determined using `cv2.CAP_PROP_FRAME_COUNT`. The code then extracts the first 300 frames and saves them as individual JPG images in a folder named "Frame".

## Part 3: Image Masking

This section of the code processes the images in the "Frame" folder by masking them using OpenCV's rectangle drawing and bitwise operations. To mask the images, a black image with the same size as the frame is created using mask = np.zeros((height, width, 3), np.uint8), and a white rectangle is drawn on the mask to cover everything outside of the middle square. The mask is then inverted and applied to the frame using the cv2.bitwise_and function.

## Part 4: Vertical Belt Length Detection

Using two edge detection algorithms - Hough and Sobel - to detect the edges in the images.

### Sobel Edge Detection

First, I extract the frames from the video and apply Sobel edge detection and Canny edge detection to the grayscale image. I then apply the Sobel edge detection on the Y-axis to detect vertical edges and save the resulting image in the "Edge" folder. The Sobel algorithm works by computing the gradient of the image using two linear operators, one for detecting horizontal edges and the other for detecting vertical edges. The output of the two convolutions is combined to create a single edge map that highlights areas of high intensity change in any direction. This edge map is thresholded to identify the edges in the original belt image.

<img src="https://raw.githubusercontent.com/wyniemko/amazon-conveyor-tracking/main/images/Sobel%20Edge%20Detection.webp" alt="Sobel Algorithm">
<br>
<p align="center">
  <em>Using the Sobel edge detection method to detect edges in an image by computing the gradient of the image. This is done using two linear operators, one for detecting horizontal edges and the other for detecting vertical edges, which are applied to the image to obtain the gradient components in both directions.</em>
</p>

### Hough Transformation
Next, I use the Hough transformation algorithm to detect the straight lines in the image. The Hough transformation algorithm works by converting the image from Cartesian coordinates (x, y) to polar coordinates (r, θ) using the Hough transform. This transforms each point in the original image to a line in the transformed image.

<img src="https://raw.githubusercontent.com/wyniemko/amazon-conveyor-tracking/main/images/Hough%20Transformation.webp" alt="Hough Transformation Algorithm">
<br>
<p align="center">
  <em>The Hough Space is now represented with ρ and θ instead of slope a and intercept b, where the horizontal axis is for the θ values and the vertical axis is for the ρ values. An edge point generates a cosine curve in the Hough Space, replacing the straight line representation and resolving the issue of unbounded values of a that arises when dealing with vertical lines.</em>
</p>

The algorithm counts the number of intersections of lines in the transformed image, which correspond to points in the original image that are part of the same line. By thresholding the number of intersections, we can identify the lines in the original belt image.

After processing the images to grayscale with Sobel edge and Canny edge detection, I loop through each file in the "Edge" folder and apply the Hough transformation algorithm to detect the vertical lines in the image. I store the resulting value of the "straightness" in a list called "straightness_values".

## Conclusion

In conclusion, the presented code offers a straightforward solution for conducting various video analysis tasks utilizing OpenCV and Numpy. With the input of conveyor belt speed and length, the code is capable of automatically computing the cycle length and extracting essential frames for further analysis. The code efficiently applies masking and edge detection techniques to identify and measure the vertical length of the gray belt, making it a valuable tool for studying conveyor belt operations.

<img src="https://raw.githubusercontent.com/wyniemko/amazon-conveyor-tracking/main/images/flowchart.png" alt="Flowchart">
<br>
<p align="center">
  <em>Flow-chart: order of operations</em>
</p>
