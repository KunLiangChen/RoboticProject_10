import cv2
import time
from robomaster import robot
from ultralytics import YOLO

def main():
    # --- Configuration ---
    # Replace 'path/to/your/best.pt' with the actual path to your trained YOLOv8 model file
    model_path = 'path/to/your/best.pt'

    # Confidence threshold for displaying detections
    conf_threshold = 0.5

    # --- Initialize YOLO Model ---
    print(f"Loading YOLO model from {model_path}...")
    try:
        model = YOLO(model_path)
        print("Model loaded successfully.")
        # Verify model configuration matches your dataset
        print(f"Model names: {model.names}") # Should contain ['block']
        print(f"Model number of classes (nc): {len(model.names)}") # Should be 1
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Initialize RoboMaster Robot ---
    ep_robot = robot.Robot()
    try:
        # Connect to the robot (adjust version if needed)
        ep_robot.initialize(conn_type="sta", sn=None) # Use "sta" for station mode, "ap" for access point mode
        print("Connected to RoboMaster robot.")

        # Get the camera module
        ep_camera = ep_robot.camera

        # Start camera video stream (default resolution is usually fine)
        ep_camera.start_video_stream(display=False) # We will display using OpenCV
        print("Camera stream started.")

        print("Starting video stream and inference. Press 'q' in the display window to quit.")

        # --- Main Loop ---
        while True:
            # Get the latest frame from the robot's camera
            frame = ep_camera.read_cv2_image(strategy="newest")
            if frame is None:
                print("Warning: Failed to grab frame.")
                time.sleep(0.01) # Brief pause before retrying
                continue

            # --- Perform YOLO Inference ---
            try:
                # The model returns a list of Results objects (one for each image/frame)
                results = model(frame, conf=conf_threshold, verbose=False) # verbose=False reduces console output

                # --- Process Results and Draw Annotations ---
                # `results[0]` corresponds to the first (and only) image/frame we passed
                annotated_frame = results[0].plot() # This draws the bounding boxes, labels, and confidence scores

            except Exception as e:
                print(f"Error during inference: {e}")
                annotated_frame = frame # Display the original frame if inference fails

            # --- Display the Result ---
            cv2.imshow("RoboMaster YOLO Inference", annotated_frame)

            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred with the robot: {e}")

    finally:
        # --- Cleanup ---
        print("Stopping video stream and closing connections...")
        try:
            ep_camera.stop_video_stream()
        except:
            pass
        try:
            ep_robot.close()
        except:
            pass
        cv2.destroyAllWindows()
        print("Cleanup complete.")

if __name__ == "__main__":
    main()