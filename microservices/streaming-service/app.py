import cv2
import pika
import pickle
from flask import Flask, Response, render_template
import threading
import time

app = Flask(__name__)

latest_info = {
    "frame": None,
    "violations": 0,
    "timestamp": ""
}

def rabbitmq_listener():
    connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmq'))
    channel = connection.channel()
    channel.queue_declare(queue="annotated_frames")

    def callback(ch, method, properties, body):
        data = pickle.loads(body)
        latest_info["frame"] = data['frame']
        latest_info["violations"] = data['number_of_violation']
        latest_info["timestamp"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data['timestamp']))

    channel.basic_consume(queue='annotated_frames', on_message_callback=callback, auto_ack=True)
    print("✅ Streaming service is listening to annotated_frames...")
    channel.start_consuming()

def generate_stream():
    while True:
        frame = latest_info["frame"]
        if frame is not None:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
            )

@app.route('/video')
def video_feed():
    return Response(generate_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def dashboard():
    return render_template("index.html",
                           violations=latest_info["violations"],
                           timestamp=latest_info["timestamp"])

if __name__ == '__main__':
    t = threading.Thread(target=rabbitmq_listener, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=8000, debug=False)
