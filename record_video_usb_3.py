import os
# Suppress Qt warning (use 'xcb' for X11 or 'offscreen' for headless)
os.environ["QT_QPA_PLATFORM"] = "xcb"
import cv2
import time
from datetime import datetime

# Configuration
VIDEO_RESOLUTION = (720, 720)
FPS = 30
OUTPUT_DIR = "/home/pi/Desktop/videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CAMERA_ID = 1
RECORDING_DURATION = 10  # seconds

def record_video():
    cap = None
    out = None
    
    try:
        # Initialize camera
        cap = cv2.VideoCapture(CAMERA_ID)
        if not cap.isOpened():
            raise RuntimeError("Error: Could not open camera!")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_RESOLUTION[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_RESOLUTION[1])
        cap.set(cv2.CAP_PROP_FPS, FPS)
        
        # Prepare video writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(OUTPUT_DIR, f"usb_video_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_file, fourcc, FPS, VIDEO_RESOLUTION)
        
        print(f"Starting recording: {output_file}")
        start_time = time.time()
        
        while (time.time() - start_time) < RECORDING_DURATION:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to grab frame!")
                continue
            
            out.write(frame)
            
            # Optional preview (press 'q' to quit early)
            cv2.imshow('Recording...', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nRecording stopped by user")
    except Exception as e:
        print(f"Error during recording: {str(e)}")
    finally:
        # Cleanup resources
        if out is not None:
            out.release()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print(f"Recording complete. Saved to: {output_file}")

if __name__ == "__main__":
    record_video()
