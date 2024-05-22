import cv2
from numpy import half
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
from queue import Queue, Empty
import signal
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

connection = psycopg2.connect(
    database="postgres", user="tushar", password="tushar", host="localhost", port=5432)

cur = connection.cursor()

model3 = YOLO('fashion.pt')
model2 = YOLO('PPE_detection.pt')
model1 = YOLO('Fire_detection.pt')
model_ = YOLO('best.pt')

violation_mqtt_topic = "violation"
broker_address = "localhost"
broker_port = 1883

zone_location_map = {
    'Zone_1': 'Zone_1',
    'Zone_2': 'Zone_2',
    'Zone_3': 'Zone_3',
    'Zone_4': 'Zone_4',
    'Zone_5': 'Zone_5',
    'Zone_10': 'Zone_10',
}

notification_queue = Queue()
threads = []

def generate_random_id(length=8):
    """Generate a random string of fixed length."""
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def create_mqtt_client():
    client = mqtt.Client(paho_enums.CallbackAPIVersion.VERSION2)
    client.on_connect = MQTT.on_connect
    client.on_publish = MQTT.on_publish
    client.connect(broker_address, broker_port, keepalive=120)
    return client

@app.route('/fire', methods=['GET'])
def start_fire_detection():
    video_path = 'dalma_400240.mp4'
    mqtt_topic = "Zone_1"
    model = model1
    violation_list = [0]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(start_inferring, video_path, mqtt_topic, model, violation_list, 'fire')
        future.result()

    return jsonify({"message": "Fire detection started"}), 200

@app.route('/ppe', methods=['GET'])
def start_ppe_detection():
    video_path = 'Hardhat.mp4'
    mqtt_topic = "Zone_4"
    model = model_
    violation_list = [2, 4]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(start_inferring, video_path, mqtt_topic, model, violation_list)
        future.result()

    return jsonify({"message": "PPE detection started"}), 200

@app.route('/ppe2', methods=['GET'])
def start_ppe2_detection():
    video_path = 'vest.mp4'
    mqtt_topic = "Zone_4"
    model = model_
    violation_list = [2, 4]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(start_inferring, video_path, mqtt_topic, model, violation_list)
        future.result()

    return jsonify({"message": "PPE detection started"}), 200

@app.route('/uniform', methods=['GET'])
def start_uniform_detection():
    video_path = 'uniform.mp4'
    mqtt_topic = "Zone_10"
    model = model3
    #{0: 'sunglass', 1: 'hat', 2: 'jacket', 3: 'shirt', 4: 'pants', 5: 'shorts', 6: 'skirt', 7: 'dress', 8: 'bag', 9: 'shoe'}
    violation_list = [3, 4, 9]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(start_inferring, video_path, mqtt_topic, model, violation_list)
        future.result()

    return jsonify({"message": "PPE detection started"}), 200

@app.route('/all', methods=['GET'])
def start_all_detection():
    video_paths = ['dalma_400240.mp4', 'Hardhat.mp4', 'vest.mp4', 'oilrig.mp4', 'ppe.mp4']
    models = [model1, model2, model2, model_, model_]
    mqtt_topics = ["Zone_1", "Zone_2", "Zone_3", "Zone_4", "Zone_5"]
    violation_lists = [[0], [0], [2], [4], [3]]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for video_path, model, mqtt_topic, violation_list in zip(video_paths, models, mqtt_topics, violation_lists):
            future = executor.submit(start_inferring, video_path, mqtt_topic, model, violation_list)
            futures.append(future)

        for future in futures:
            future.result()

    return jsonify({"message": "All detections started"}), 200

def send_notification(stop_event):
    while not stop_event.is_set():
        try:
            frame, title, mqtt_topic, confidence, result, filtered_boxes = notification_queue.get(timeout=1)
            if filtered_boxes:
                result.boxes = filtered_boxes
                annotated_frame = result.plot()
            else:
                annotated_frame = frame

            (flag, img) = cv2.imencode('.jpg', annotated_frame)
            jpg_as_text = base64.b64encode(img)
            ts = datetime.datetime.now()
            client = create_mqtt_client()
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
        except Empty:
            continue
        except Exception as e:
            logger.error(f"Error in send_notification: {e}")

def start_inferring(video_path, mqtt_topic, model, violation_list, type=None):
    thread_id = generate_random_id()
    threading.current_thread().name = f"Thread-{mqtt_topic}-{thread_id}"

    consolidated_violations = []
    no_violation_counter = 0
    violation_states = {}
    client = create_mqtt_client()  

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
                frame_violations = []

                for res in results:
                    if res.boxes:
                        for box in res.boxes:
                            violation_class = box.cls.cpu().item()
                            confidence = box.conf.cpu().item()
                            if violation_class in violation_list and confidence >= 0.2:
                                frame_violations.append({
                                    'violation_class': violation_class,
                                    'confidence': confidence,
                                    'box': box
                                })
                    
                violation_counts = {}
                for violation in frame_violations:
                    cls_name = res.names[violation['violation_class']]
                    if cls_name not in violation_counts:
                        violation_counts[cls_name] = 0
                    violation_counts[cls_name] += 1

                
                violation_info = [f"|{cls_name}| C-{count}" for cls_name, count in violation_counts.items()]

                for info in violation_info:
                    print(info)

                if frame_violations:
                    consolidated_violations.append(frame_violations)
                    no_violation_counter = 0

                    if len(consolidated_violations) >= 1:
                        consolidated_report = []
                        for violations in consolidated_violations:
                            for violation in violations:
                                consolidated_report.append(violation)

                        highest_conf_violation = max(consolidated_report, key=lambda x: x['confidence'])
                        violation_class = highest_conf_violation['violation_class']
                        confidence = highest_conf_violation['confidence']
                        filtered_boxes = [v['box'] for v in consolidated_report]

                        if violation_class not in violation_states:
                            violation_states[violation_class] = {'active': False, 'last_confidence': 0.0, 'count': 0,'timestamp': datetime.datetime.min}

                        current_state = violation_states[violation_class]
                        current_state['count'] += 1
                        current_time = datetime.datetime.now()
                        time_since_first_violation = (current_time - current_state['timestamp']).total_seconds()

                        if not current_state['active'] and confidence > 0.5:
                            current_state['active'] = True
                            current_state['timestamp'] = current_time
                            notification_queue.put((frame, f'Started violation {results[0].names[violation_class]} with Confi {round(confidence, 2)} {str(violation_info)}' , mqtt_topic, confidence, results[0], filtered_boxes))
                        elif current_state['active'] and confidence < 0.30 and time_since_first_violation >= 5  :
                            current_state['active'] = False
                            notification_queue.put((frame, f'Stopped violation {results[0].names[violation_class]} with Confi {round(confidence, 2)}', mqtt_topic, confidence, results[0], filtered_boxes))

                        current_state['last_confidence'] = confidence

                        if len(consolidated_violations) >= 5:
                            consolidated_violations = []
                else:
                    no_violation_counter += 1
                    if no_violation_counter >= 24:
                        notification_queue.put((frame, f'No violations detected in the last {no_violation_counter} frames', mqtt_topic, 0, results[0], []))
                        no_violation_counter = 0

                if frame_violations:
                    results[0].boxes = filtered_boxes
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

def keep_alive(stop_event):
    while not stop_event.is_set():
        client = create_mqtt_client()  
        client.publish("keepalive", "ping")
        time.sleep(30)

def signal_handler(sig, frame):
    logger.info('Signal handler called with signal', sig)
    stop_event.set()
    for thread in threads:
        thread.join()
    sys.exit(0)

if __name__ == '__main__':
    stop_event = threading.Event()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    keep_alive_thread = threading.Thread(target=keep_alive, args=(stop_event,))
    keep_alive_thread.daemon = True
    keep_alive_thread.start()
    threads.append(keep_alive_thread)

    notification_thread = threading.Thread(target=send_notification, args=(stop_event,))
    notification_thread.daemon = True
    notification_thread.start()
    threads.append(notification_thread)

    app.run(host='0.0.0.0', port=8000)
