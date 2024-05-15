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

# Database connection
connection = psycopg2.connect(
    database="postgres", user="tushar", password="tushar", host="localhost", port=5432)
cur = connection.cursor()

# Load the YOLOv8 models
model = YOLO('PPE_detection.pt')
model1 = YOLO('Fire_detection.pt')

violation_list = [2.0, 3.0, 4.0]
violation_mqtt_topic = "violation"
broker_address = "localhost"
broker_port = 1883

# MQTT client setup
client = mqtt.Client(paho_enums.CallbackAPIVersion.VERSION2)
client.on_connect = MQTT.on_connect  # Use the class method as a callback
client.on_publish = MQTT.on_publish  # Use the class method as a callback
client.connect(broker_address, broker_port, keepalive=60)

zone_location_map = {
    'Zone_1': 'Zone_1',
    'Zone_2': 'Zone_2',
    'Zone_3': 'Zone_3'
}


notification_interval = 10  
last_notification_time = datetime.datetime.now() - datetime.timedelta(seconds=notification_interval)


def start_inferring(video_path, mqtt_topic, model):
    global cur, last_notification_time
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        print("CAP", cap.isOpened())

        while cap.isOpened():
            success, frame = cap.read()
            if success:
                frame = cv2.resize(frame, (640, 480))
                results = model(frame)

                current_time = datetime.datetime.now()
                if (current_time - last_notification_time).total_seconds() >= notification_interval:
                    for res in results:
                        if res.boxes:
                            for i, box in enumerate(res.boxes):
                                confidence = box.conf.cpu().item()
                                if confidence > 0.50:
                                    (_flag, _img) = cv2.imencode('.jpg', frame)
                                    _jpg_as_text = base64.b64encode(_img)
                                    ts = current_time
                                    client.publish(violation_mqtt_topic, json.dumps({
                                        'thumbnail': _jpg_as_text.decode('utf-8'),
                                        'ts': ts.strftime("%Y-%m-%dT%H:%M:%S"),
                                        'location': zone_location_map.get(mqtt_topic, 'Unknown Location'),
                                        'title': res.names[i]
                                    }))
                                    query = """
                                        INSERT INTO violations (incident_name, time, location)
                                        VALUES (%(incident_name)s, %(time)s, %(location)s)
                                    """
                                    cur.execute(query, {
                                        'incident_name': res.names[i],
                                        'time': ts,
                                        'location': mqtt_topic
                                    })
                                    connection.commit()

                    if any(element in res.boxes.cls.cpu().tolist() for element in violation_list):
                        (_flag, _img) = cv2.imencode('.jpg', frame)
                        _jpg_as_text = base64.b64encode(_img)
                        ts = current_time
                        client.publish(violation_mqtt_topic, json.dumps({
                            'thumbnail': _jpg_as_text.decode('utf-8'),
                            'ts': ts.strftime("%Y-%m-%dT%H:%M:%S"),
                            'location': zone_location_map.get(mqtt_topic, 'Unknown Location')
                        }))

                    last_notification_time = current_time

                annotated_frame = results[0].plot()
                (flag, img) = cv2.imencode('.jpg', annotated_frame)
                jpg_as_text = base64.b64encode(img)
                client.publish(mqtt_topic, jpg_as_text)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        cap.release()
    except Exception as e:
        print(f"Exception for {mqtt_topic}", e)


stream_1_thread = threading.Thread(target=start_inferring, args=("ppe.mp4", "Zone_1", model))
stream_2_thread = threading.Thread(target=start_inferring, args=("dalma_400240.mp4", "Zone_2", model1))
stream_3_thread = threading.Thread(target=start_inferring, args=("fire.mp4", "Zone_3", model1))

stream_1_thread.start()
stream_2_thread.start()
stream_3_thread.start()

stream_1_thread.join()
stream_2_thread.join()
stream_3_thread.join()
