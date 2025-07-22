import sqlite3 
import os 
from datetime import datetime
import cv2

DB_PATH = "violations.db"
def init_db():
     conn = sqlite3.connect(DB_PATH) 
     cursor = conn.cursor() 
     cursor.execute(''' CREATE TABLE IF NOT EXISTS violations ( 
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    frame_path TEXT, 
                    timestamp TEXT, 
                    hand_id INTEGER, 
                    violation_label TEXT,
                     bounding_box TEXT ) '''
                    ) 
     conn.commit() 
     conn.close()

def save_violation(frame, hand_id, bbox, output_dir="violations_frames"): 
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    frame_name = f"violation_{timestamp}_hand{hand_id}.jpg"
    frame_path = os.path.join(output_dir, frame_name)

    cv2.imwrite(frame_path, frame)


    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO violations (frame_path, timestamp, hand_id, violation_label, bounding_box)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        frame_path,
        timestamp,
        hand_id,
        "missing_scooper",
        str(bbox)
    ))
    conn.commit()
    conn.close()