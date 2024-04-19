import psycopg2
import json
import subprocess

# Connect to PostgreSQL
conn = psycopg2.connect(
    database="postgres",
    user="postgres",
    password="postgres",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

# Mosquitto subscription command
mosquitto_sub_command = [
    "mosquitto_sub",
    "-h", "localhost",
    "-p", "1883",
    "-t", "Zone_1",
    "-v"  # Verbose mode to print messages
]

# Start Mosquitto subscriber subprocess
subprocess_sub = subprocess.Popen(mosquitto_sub_command, stdout=subprocess.PIPE)
# Consume and process messages
try:    
    for message in subprocess_sub.stdout:        
        # Assuming messages are in JSON format
        try:
            data = message.decode("utf-8")            
            data = data.split(" ")
            print("data------------------", [float(item.strip('[],\n')) for item in data[1:]])
                 
            # Insert data into PostgreSQL table
            sql = "INSERT INTO mqtt_messages (topic, payload) VALUES (%s, %s);"
            values = (data[0],[float(item.strip('[],\n')) for item in data[1:]])  # Replace with your actual data fields
            cursor.execute(sql, values)
            conn.commit()

            print("Message inserted into PostgreSQL:", data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
except KeyboardInterrupt:
    # Handle keyboard interrupt (Ctrl+C) to gracefully exit the script
    print("Script interrupted. Closing connection.")
finally:
    # Clean up resources
    cursor.close()
    conn.close()
    subprocess_sub.terminate()
