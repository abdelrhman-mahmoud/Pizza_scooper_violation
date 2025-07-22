import cv2
from ultralytics import YOLO
import pika
import pickle
import time

class VideoProcessor:
    def __init__(self, yolo_path, video_path):
        self.model = YOLO(yolo_path)
        self.video_path = video_path
        self.connection = None
        self.channel = None

    def connect_rabbitmq(self):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmq'))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue="detected_objects")

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            return

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            results = self.model.track(source=frame, persist=True, tracker="bytetrack.yaml")[0]
            
            detected_objects_data = []
            if results.boxes is not None and results.boxes.id is not None:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    label = results.names[cls_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    box_coords = (x1, y1, w, h)
                    object_id = int(box.id[0])
                    
                    detected_objects_data.append({
                        'id': object_id,
                        'label': label,
                        'box': box_coords
                    })
            
            send_data = {
                'frame_number': frame_count,
                'detected_objects': detected_objects_data,
                'frame': frame # Sending the frame for visualization in streaming service
            }
            send_body = pickle.dumps(send_data)
            self.channel.basic_publish(exchange='', routing_key='detected_objects', body=send_body)

        cap.release()
        if self.connection:
            self.connection.close()
        print("Video processing finished.")

if __name__ == '__main__':
    processor = VideoProcessor(
        yolo_path="models/yolo12/best.pt", # Assuming models are directly in detection-service
        video_path="/dataset/3.mp4" # This path needs to be accessible within the container
    )
    processor.connect_rabbitmq()
    processor.process_video()