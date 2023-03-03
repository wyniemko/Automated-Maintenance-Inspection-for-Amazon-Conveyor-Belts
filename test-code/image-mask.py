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
