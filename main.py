from typing import Any
from ultralytics import YOLO
import cv2 
import math 
from deep_sort_realtime.deepsort_tracker import DeepSort
import easyocr
import os
import string

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class LicensePlateDetector():
    def __init__(self, capture, output_name):
        self.capture = capture
        # Load YOLO model from ultralytics
        self.model = self.load_model()
        # Get class names from the YOLO model
        self.CLASS_NAMES_DICT = self.model.model.names
        # Initialize an EasyOCR reader for reading text from license plates
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.translator = str.maketrans('', '', string.punctuation)
        self.output_name = output_name
        self.prev_box_size = None

    def load_model(self):
        model = YOLO("model.pt")
        model.fuse()
        return model

    def predict(self, img):
        results = self.model(img, stream=True)
        return results

    def plot_boxes(self, results, img):
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cls = int(box.cls[0])
                currentClass = self.CLASS_NAMES_DICT[cls]
                conf = math.ceil(box.conf[0] * 100) / 100

                # Only consider objects with confidence greater than 0.3
                if conf > 0.3:
                    detections.append((([x1, y1, w, h]), conf, currentClass))

        return detections, img

    def track_detect(self, detections, img, tracker):
        # Use the DeepSort tracker to associate and track detected objects
        tracks = tracker.update_tracks(detections, frame=img)

        for track in tracks:
            if not track.is_confirmed():
                continue
            bbox = track.to_ltrb()
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            if x1 >= 0 and y1 >= 0 and x2 <= img.shape[1] and y2 <= img.shape[0]:
                # Read text from the region of interest in the image
                text = self.reader.readtext(img[y1:y2, x1:x2], detail=0)
                if len(text) > 0:
                    # Remove punctuation and convert text to uppercase
                    text = ' '.join(text).translate(self.translator).replace(' ', '').upper()
                else:
                    text = ''
            else:
                text = ''

            # Draw bounding boxes, text, and background for the detected objects
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(img, (x1 - 2, y1 - text_height - 12), (x1 + text_width, y1 + text_height - 30),
                          color=(255, 255, 255), thickness=cv2.FILLED)
            cv2.rectangle(img, (x1 - 2, y1 - text_height - 12), (x1 + text_width, y1 + text_height - 30),
                          color=(0, 0, 0), thickness=1)
            cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)

        return img

    def __call__(self):
        cap = cv2.VideoCapture(self.capture)
        assert cap.isOpened()

        tracker = DeepSort(max_age=10, n_init=1, nms_max_overlap=1.0, max_cosine_distance=0.3)

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        out = cv2.VideoWriter(self.output_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if ret == True:
                # Make predictions using YOLO and track objects in the frame
                results = self.predict(frame)
                detections, frames = self.plot_boxes(results, frame)
                detect_frame = self.track_detect(detections, frames, tracker)
                out.write(detect_frame)
                # Uncomment the next line if you want to display the video while processing
                # cv2.imshow('Video', detect_frame)
                # if cv2.waitKey(1) == ord('q'): break
            else:
                break

        cap.release()
        out.release()

detector = LicensePlateDetector(capture='video/video.mp4', output_name='video/video_with_tracker.avi')
detector()