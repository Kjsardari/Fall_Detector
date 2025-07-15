import cv2
import numpy as np
import tensorflow as tf
import requests
import time

# === Telegram Configuration ===
BOT_TOKEN = '{Your Telegram Bot Token}'
CHAT_ID = '{Your Telegram Chat ID}'

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            print("Failed to send Telegram message:", response.text)
    except Exception as e:
        print("⚠️ Error sending message:", e)

# === Load the trained model ===
model = tf.keras.models.load_model('model/fall_detector.h5')

# === Start webcam ===
cap = cv2.VideoCapture(1)

fall_start_time = None
fall_alert_sent = False
FALL_DURATION_THRESHOLD = 7  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and normalize
    input_frame = cv2.resize(frame, (128, 128)) 
    input_frame = input_frame / 255.0
    input_frame = np.expand_dims(input_frame, axis=0)

    # Model prediction
    prediction = model.predict(input_frame)[0][0]
    label = "Fall" if prediction > 0.5 else "Normal"
    color = (0, 0, 255) if label == "Fall" else (0, 255, 0)

    # Show label on frame
    cv2.putText(frame, f"Status: {label}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.imshow("Fall Detection", frame)

    # Timer logic
    current_time = time.time()
    if label == "Fall":
        if fall_start_time is None:
            fall_start_time = current_time
        elif (current_time - fall_start_time) >= FALL_DURATION_THRESHOLD and not fall_alert_sent:
            print("🚨 Fall Detected for 7+ seconds! Sending alert...")
            send_telegram_message("🚨 Fall Detected for over 7 seconds! Please check urgently.")
            fall_alert_sent = True
    else:
        fall_start_time = None
        fall_alert_sent = False

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
