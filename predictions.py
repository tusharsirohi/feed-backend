import cv2
from ultralytics import YOLO
from notification import MQTT
import paho.mqtt.client as mqtt
import paho.mqtt.enums as paho_enums
import datetime
import base64
import json
import threading
# Load the YOLOv8 model
model = YOLO(
    'C:\\Users\\mofizul.islam\\Downloads\\PPE_Detection\\TrainedModelWeightsFile\\best.pt')
violation_list = [2.0, 3.0, 4.0]
violation_mqtt_topic = "violation"
broker_address = "localhost"
broker_port = 1883

client = mqtt.Client(paho_enums.CallbackAPIVersion.VERSION2)
client.on_connect = MQTT.on_connect  # Use the class method as a callback
client.on_publish = MQTT.on_publish  # Use the class method as a callback
client.connect(broker_address, broker_port, keepalive=60)

zone_location_map = {
    'Zone_1': 'Zone 1',
    'Zone_2': 'Some Other Zone'
}


def start_inferring(video_path, mqtt_topic):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            # frame = cv2.resize(frame, (640, 480))
            frame = cv2.resize(frame, (320, 240))
            results = model(frame)

            print("00000000000000000000000")
            result = results[0].boxes.cls.cpu().tolist()
            if any(element in result for element in violation_list):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"violation_frame_{timestamp}.jpg"
                cv2.imwrite(filename, results[0].plot())
                frame_thumbnail = cv2.resize(frame, (160, 120))
                (_flag, _img) = cv2.imencode('.jpg', frame_thumbnail)
                _jpg_as_text = base64.b64encode(_img)
                client.publish(violation_mqtt_topic, json.dumps({
                    'thumbnail': _jpg_as_text.decode('utf-8'),
                    'ts': datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                    'location': zone_location_map.get(mqtt_topic, 'Unknown Location')
                }))
            annotated_frame = results[0].plot()

            # Display the annotated frame
            # cv2.imshow("YOLOv8 Inference", annotated_frame)
            (flag, img) = cv2.imencode('.jpg', annotated_frame)
            jpg_as_text = base64.b64encode(img)
            client.publish(mqtt_topic, jpg_as_text)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    cap.release()


stream_1_thread = threading.Thread(target=start_inferring, args=(
    "C:\\Users\\mofizul.islam\\Downloads\\PPE_Detection\\InputDemoVideo\\demo3.mp4", "Zone_2"))

stream_2_thread = threading.Thread(target=start_inferring, args=(
    "C:\\Users\\mofizul.islam\\Downloads\\PPE_Detection\\InputDemoVideo\\demo.mp4", "Zone_1"))

stream_1_thread.start()
stream_2_thread.start()
# Release the video capture object and close the display window
# cv2.destroyAllWindows()

stream_1_thread.join()
stream_2_thread.join()
