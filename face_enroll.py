import cv2
import os

def enroll_new_person(name, save_dir='dataset', num_samples=10):
    cap = cv2.VideoCapture(0)
    person_dir = os.path.join(save_dir, name)

    if not os.path.exists(person_dir):
        os.makedirs(person_dir)

    print(f"Capturing {num_samples} samples for {name} 📸...")

    count = 0
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            continue

        cv2.imshow('Capture Face', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            img_path = os.path.join(person_dir, f"{count}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"Saved {img_path}")
            count += 1

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Enrollment for {name} completed ✅")
