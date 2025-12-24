import cv2
import time
import numpy as np
import os

# Create a directory to save the burst images
output_dir = "burst_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def capture_burst(num_images=5):
    """
    Captures a burst of images from the webcam.
    """
    # Open the default camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print(f"Camera opened successfully. Press 's' to capture a {num_images}-photo burst, 'q' to quit.")

    # List to store captured images
    burst_frames = []

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the live feed
        cv2.imshow("Webcam Feed", frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            # Quit the application
            break
        elif key == ord('s'):
            # Capture a burst
            print("starting")
            time.sleep(5) 
            print(f"Capturing {num_images} images...")
            for i in range(num_images):
                # Ensure the frame is fresh
                _, burst_frame = cap.read()
                if burst_frame is not None:
                    burst_frames.append(burst_frame)
                    # Small delay to allow some variation in frames, though they may still be very similar
                    time.sleep(0.1) 
                else:
                    print(f"Warning: Could not capture image {i+1}.")
            print(f"Captured {len(burst_frames)} images.")
            
            # Save the captured images
            timestamp = int(time.time())
            for idx, img in enumerate(burst_frames):
                img_name = os.path.join(output_dir, f"burst_{timestamp}_{idx+1}.png")
                cv2.imwrite(img_name, img)
                print(f"Saved {img_name}")
            
            # Clear the burst list for the next capture
            burst_frames = []
            print("Ready for next burst.")

    # Release the camera and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # You can change the number of images in the burst here
    capture_burst(num_images=10)
