# Amazon Conveyor Belt Tracking

This code uses OpenCV and Numpy to process a video and perform various operations on it, including recording, frame extraction, masking, and contour detection.

## Requirements

- Python 3
- OpenCV
- Numpy

## Part 1: Taking the Video

In the first part of the code, the user is prompted to enter the speed and length of a conveyor belt. The cycle length of the belt is then calculated using the formula:

cycle_length = length / speed * 5280 / 3600


The video recording is started using OpenCV's `cv2.VideoCapture` function and the video is saved in MP4 format. The recording stops after the calculated cycle length has elapsed.

## Part 2: Extracting Frames from the Video

In this part, the video file is loaded using `cv2.VideoCapture` and the number of frames in the video is determined using `cv2.CAP_PROP_FRAME_COUNT`. The code then extracts the first 300 frames and saves them as individual JPG images in a folder named "Frame".

## Part 3: Masking the Image

In this part, the images in the "Frame" folder are masked using OpenCV's rectangle drawing and bitwise operations. The code creates a black image with the same size as the frame, then draws white rectangles on it to cover everything outside of the middle square. The result is a masked image with a clear middle square.

## Part 4: Detecting the Vertical Length of the Gray Belt

In the final part of the code, the code performs background subtraction on the masked images in the "Frame" folder, then finds contours in the resulting images. The sum of white pixels in each contour is calculated and stored in a list. The contour with the largest sum of white pixels is assumed to be the contour of the gray belt, and its vertical length is calculated based on its height.

## Conclusion

This code provides a simple way to perform various video analysis operations using OpenCV and Numpy. By entering the speed and length of the conveyor belt, the code can automatically calculate the cycle length and extract the necessary frames for analysis. The code can then mask the frames and detect the vertical length of the gray belt.
