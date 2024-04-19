import paho.mqtt.client as mqtt


class MQTT:
    @staticmethod
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker")
            client.publish("Zone_1", userdata)
        else:
            print("Connection failed with error code {}".format(rc))

    @staticmethod
    def on_publish(client, userdata, mid, f, fv):
        print("Message published", f, fv)
