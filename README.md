# Amazon Conveyor Belt Tracking

This code processes a video to detect the straightness of edges in the frames. It does this by extracting frames from the video, masking the images, detecting the edges of the images, and testing the straightness of the edges.

## Requirements

- Python 3
- OpenCV
- Numpy

## OpenCV Functions

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

## Part 1: Taking the Video

In the first part of the code, the user is prompted to enter the speed and length of a conveyor belt. The cycle length of the belt is then calculated using the formula:

cycle_length = length / speed * 5280 / 3600

The video codec used to compress the video file is set to mp4v using the cv2.VideoWriter_fourcc function. The total number of frames in the video is obtained using the cv2.VideoCapture.get function with the cv2.CAP_PROP_FRAME_COUNT argument. The images are read and written to files using the cv2.imread and cv2.imwrite functions, respectively. The recording stops after the calculated cycle length has elapsed.

## Part 2: Extracting Frames from the Video

In this part, the video file is loaded using `cv2.VideoCapture` and the number of frames in the video is determined using `cv2.CAP_PROP_FRAME_COUNT`. The code then extracts the first 300 frames and saves them as individual JPG images in a folder named "Frame".

## Part 3: Masking the Image

In this part, the images in the "Frame" folder are masked using OpenCV's rectangle drawing and bitwise operations. To mask the images, a black image with the same size as the frame is created (mask = np.zeros((height, width, 3), np.uint8)), and a white rectangle is drawn on the mask to cover everything outside of the middle square. The mask is then inverted and applied to the frame using the cv2.bitwise_and function.

## Part 4: Detecting the Vertical Length of the Gray Belt

The Sobel and Canny edge detection algorithms are used to detect edges in the images. 

The Hough transformation algorithm is a technique used for detecting shapes in an image, specifically lines and curves. The algorithm works by converting the image from Cartesian coordinates (x, y) to polar coordinates (r, θ) using the Hough transform. This transforms each point in the original image to a line in the transformed image.

<img src="https://raw.githubusercontent.com/wyniemko/conveyor-belt-tracking/main/1_Cr73Mte5NNgO16D4moKDQg.webp" alt="Hough Transformation Algorithim">
<br>
<p align="center">
  <em>Gradient of f(x, y) at co-ordinate (x, y) is defined as 2 dimensional column vectors pointing to direction of greatest rate of change f at that location.</em>
</p>

The algorithm then counts the number of intersections of lines in the transformed image, which correspond to points in the original image that are part of the same line. By thresholding the number of intersections, we can identify the lines in the original belt image.

The Sobel edge detection algorithm is a technique used to detect edges in an image. The algorithm works by convolving the image with two filters, one for the x-direction and one for the y-direction. These filters highlight areas in the image where there is a sharp change in intensity in the x or y direction, respectively.

<img src="https://raw.githubusercontent.com/wyniemko/conveyor-belt-tracking/main/1_kB-_G3KdXA7r5v403EbwEg.webp" alt="Sobel Algorithim">
<br>
<p align="center">
  <em>The equation to calculate a slope of a line.</em>
</p>

The output of the two convolutions are then combined to create a single edge map that highlights areas of high intensity change in any direction. This edge map are used a thresholded to identify the edges in the original belt image.

## Conclusion

This code provides a simple way to perform various video analysis operations using OpenCV and Numpy. By entering the speed and length of the conveyor belt, the code can automatically calculate the cycle length and extract the necessary frames for analysis. The code can then mask the frames and detect the vertical length of the gray belt.
