import open3d as o3d
import numpy as np
import cv2
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

# Define the intrinsic parameters of the camera
fx = 1000
fy = 1000
cx = 320
cy = 240

# Create an empty Point Cloud object
pcd = o3d.geometry.PointCloud()

# Loop through all the frames in the video
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create a Point Cloud from the depth values in the grayscale image
        depth = np.float32(gray)
        depth /= 255.0
        depth = cv2.medianBlur(depth, 5)
        points = np.zeros((480, 640, 3))
        points[:,:,0] = np.arange(640).reshape(1, -1).repeat(480, 0)
        points[:,:,1] = np.arange(480).reshape(-1, 1).repeat(640, 1)
        points[:,:,2] = depth
        points = points.reshape(-1, 3)
        points = points[points[:,2] > 0]

        # Convert the Point Cloud to the camera coordinate system
        points[:,0] = (points[:,0] - cx) * points[:,2] / fx
        points[:,1] = (points[:,1] - cy) * points[:,2] / fy

        # Add the Point Cloud to the final Point Cloud model
        pcd.points = o3d.utility.Vector3dVector(np.concatenate((pcd.points, points)))

        # Display the Point Cloud
        o3d.visualization.draw_geometries([pcd])

        if o3d.vis


        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Save the final Point Cloud model to a file
o3d.io.write_point_cloud('point_cloud.ply', pcd)

# Release the video file and destroy all windows
cap.release()
cv2.destroyAllWindows()
