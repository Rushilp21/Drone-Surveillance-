import cv2
import mediapipe as mp
import time
import config  # Import settings from config.py

class PrivacyStream:
    def __init__(self):
        # Initialize MediaPipe Face Detection (Lightweight & Fast)
        # Force Python to find the internal module
        import mediapipe.python.solutions.face_detection as mp_face_detection
        self.face_detection = mp_face_detection.FaceDetection(
            min_detection_confidence=config.DETECTION_CONFIDENCE
        )
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=config.DETECTION_CONFIDENCE
        )
        
        # Initialize Video Capture
        self.cap = cv2.VideoCapture(config.CAMERA_SOURCE)
        
        # Optimization: Set Camera Resolution (Hardware level if supported)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    def apply_privacy_filter(self, frame):
        """
        Detects faces and applies a heavy Gaussian blur to the ROI (Region of Interest).
        """
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame for face detection
        results = self.face_detection.process(frame_rgb)
        
        if results.detections:
            h, w, _ = frame.shape
            
            for detection in results.detections:
                # Get Bounding Box Info
                bboxC = detection.location_data.relative_bounding_box
                
                # Calculate pixel coordinates
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)

                # Clamp coordinates to frame boundaries (prevent crashes)
                x = max(0, x)
                y = max(0, y)
                width = min(w - x, width)
                height = min(h - y, height)

                # Extract the Face Region of Interest (ROI)
                face_roi = frame[y:y+height, x:x+width]

                # Apply Gaussian Blur to the ROI
                # Note: We only blur if the area is valid
                if face_roi.size > 0:
                    blurred_face = cv2.GaussianBlur(face_roi, config.BLUR_INTENSITY, 0)
                    
                    # Put the blurred face back into the original frame
                    frame[y:y+height, x:x+width] = blurred_face
                    
                    # Optional: Draw a green border to show "Protected" status
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    cv2.putText(frame, "PRIVACY PROTECTED", (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame

    def run(self):
        print(f"Starting SkyGuard Feed... Press 'q' to exit.")
        prev_time = 0

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame or video ended.")
                break

            # 1. Resize Frame (Software level) for consistent processing speed
            frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

            # 2. Apply Privacy Filter (if enabled in config)
            if config.ENABLE_BLUR:
                frame = self.apply_privacy_filter(frame)

            # Calculate and Display FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # 3. Display the Output
            cv2.imshow('SkyGuard Input Layer (Privacy Mode)', frame)

            # Exit condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    stream = PrivacyStream()
    stream.run()