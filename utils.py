import os
import cv2
import csv
from datetime import datetime

VISITOR_DIR = 'visitors'
LOG_FILE = 'logs/visitor_log.csv'

# Create folders if missing
os.makedirs(VISITOR_DIR, exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Save Image
def save_visitor(identity, frame):
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename = f"{VISITOR_DIR}/{identity}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)

    # Log entry
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([identity, timestamp])

# Current Timestamp
def timestamp_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
