import cv2
from ultralytics import YOLO
from notification import MQTT
import paho.mqtt.client as mqtt
import paho.mqtt.enums as paho_enums
import datetime
import base64
import json
import threading
import psycopg2
from flask import Flask, jsonify
from flask_cors import CORS
import concurrent.futures
import random
import string
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app) 

connection = psycopg2.connect(
    database="postgres", user="tushar", password="tushar", host="localhost", port=5432)

cur = connection.cursor()

model = YOLO('PPE_detection.pt')
model1 = YOLO('Fire_detection.pt')
model_ = YOLO('best.pt')

violation_list_ = [2, 4]
violation_mqtt_topic = "violation"
broker_address = "localhost"
broker_port = 1883

client = mqtt.Client(paho_enums.CallbackAPIVersion.VERSION2)
client.on_connect = MQTT.on_connect  
client.on_publish = MQTT.on_publish  
client.connect(broker_address, broker_port, keepalive=120)

zone_location_map = {
    'Zone_1': 'Zone_1',
    'Zone_2': 'Zone_2',
    'Zone_3': 'Zone_3'
}

violation_states = {}

def generate_random_id(length=8):
    """Generate a random string of fixed length."""
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

@app.route('/fire', methods=['GET'])
def start_fire_detection():
    video_path = 'dalma_400240.mp4'
    mqtt_topic = "Zone_2"
    model = model1

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(start_inferring, video_path, mqtt_topic, model, 'fire')
        future.result()

    return jsonify({"message": "Fire detection started"}), 200

@app.route('/ppe', methods=['GET'])
def start_ppe_detection():
    video_path = 'Construction.mp4'
    mqtt_topic = "Zone_1"
    model = model_

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(start_inferring, video_path, mqtt_topic, model)
        future.result()

    return jsonify({"message": "PPE detection started"}), 200

@app.route('/all', methods=['GET'])
def start_all_detection():
    video_paths = ['Construction.mp4', 'dalma_400240.mp4', 'ppe.mp4']
    models = [model_, model1, model_]
    mqtt_topics = ["Zone_1", "Zone_2", "Zone_3"]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for video_path, model, mqtt_topic in zip(video_paths, models, mqtt_topics):
            type_param = 'fire' if mqtt_topic == "Zone_2" else None
            future = executor.submit(start_inferring, video_path, mqtt_topic, model, type_param)
            futures.append(future)

        for future in futures:
            future.result()

    return jsonify({"message": "All detections started"}), 200

def start_inferring(video_path, mqtt_topic, model, type=None):
    global cur, violation_states
    thread_id = generate_random_id()
    threading.current_thread().name = f"Thread-{mqtt_topic}-{thread_id}"

    if type == 'fire':
        violation_list = [0]
    else:
        violation_list = violation_list_

    try:
        cap = cv2.VideoCapture(video_path)
        iteration_counter = 0

        logger.info(f"Started inference for {mqtt_topic} with thread ID {thread_id}")

        while cap.isOpened():
            success, frame = cap.read()
            iteration_counter += 1

            if success:
                frame = cv2.resize(frame, (640, 480))

                if iteration_counter % 10 != 0:
                    continue

                results = model(frame)

                filtered_boxes = []
                for res in results:
                    if res.boxes:
                        for box in res.boxes:
                            violation_class = box.cls.cpu().item()
                            confidence = box.conf.cpu().item()
                            if violation_class in violation_list:
                                filtered_boxes.append(box)

                                if violation_class not in violation_states:
                                    violation_states[violation_class] = {'active': False, 'last_confidence': 0.0}

                                current_state = violation_states[violation_class]

                                if not current_state['active'] and confidence > 0.5:
                                    current_state['active'] = True
                                    send_notification(frame, 'Started violation ' + res.names[violation_class] + ' with Confi ' + str(round(confidence, 2)), mqtt_topic, confidence, results[0])
                                elif current_state['active'] and confidence < 0.3:
                                    current_state['active'] = False
                                    send_notification(frame, 'Stopped violation ' + res.names[violation_class] + ' with Confi ' + str(round(confidence, 2)), mqtt_topic, confidence, results[0])

                                current_state['last_confidence'] = confidence

                if filtered_boxes:
                    results[0].boxes = filtered_boxes
                    filtered_boxes = []

                    annotated_frame = results[0].plot()
                else:
                    annotated_frame = frame

                (flag, img) = cv2.imencode('.jpg', annotated_frame)
                jpg_as_text = base64.b64encode(img)
                client.publish(mqtt_topic, jpg_as_text)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        cap.release()
    except Exception as e:
        logger.error(f"Exception for {mqtt_topic} with thread ID {thread_id}: {e}")

def send_notification(frame, title, mqtt_topic, confidence, result):
    (flag, img) = cv2.imencode('.jpg', result.plot() if result.boxes else frame)
    jpg_as_text = base64.b64encode(img)
    ts = datetime.datetime.now()
    client.publish(violation_mqtt_topic, json.dumps({
        'thumbnail': jpg_as_text.decode('utf-8'),
        'ts': ts.strftime("%Y-%m-%dT%H:%M:%S"),
        'location': zone_location_map.get(mqtt_topic, 'Unknown Location'),
        'title': title
    }))
    query = """
        INSERT INTO violations (incident_name, time, location)
        VALUES (%(incident_name)s, %(time)s, %(location)s)
    """
    cur.execute(query, {
        'incident_name': title,
        'time': ts,
        'location': mqtt_topic
    })
    connection.commit()

def keep_alive():
    while True:
        client.publish("keepalive", "ping")
        time.sleep(30)

if __name__ == '__main__':
    keep_alive_thread = threading.Thread(target=keep_alive)
    keep_alive_thread.daemon = True
    keep_alive_thread.start()
    app.run(host='0.0.0.0', port=8000)