from ultralytics import YOLO
import cv2
import os
import re

# ---------------- LOAD MODEL ----------------
plate_model = YOLO("./models/license_plate_detector.pt")

# ---------------- PATHS ----------------
input_root = "./outputs"  # Root input folder containing image_name folders
valid_ext = (".jpg", ".jpeg", ".png", ".bmp")

# Pattern to match valid vehicle folders
vehicle_pattern = re.compile(r"^(car|bus|truck)_\d+$", re.IGNORECASE)

# ---------------- PROCESS ----------------
for image_name in os.listdir(input_root):
    image_path = os.path.join(input_root, image_name)

    if not os.path.isdir(image_path):
        continue  # skip files

    # Iterate over only valid vehicle folders
    for vehicle_folder_name in os.listdir(image_path):
        if not vehicle_pattern.match(vehicle_folder_name):
            # print("non-matching folders")
            continue  # skip non-matching folders

        vehicle_folder = os.path.join(image_path, vehicle_folder_name)

        crop_path = os.path.join(vehicle_folder, "crop.jpg")
        if not os.path.exists(crop_path):
            print(f"crop.jpg not found in {vehicle_folder}")
            continue

        # Read crop image
        vehicle_img = cv2.imread(crop_path)
        if vehicle_img is None:
            print(f"Failed to read {crop_path}")
            continue

        # ---------------- PLATE DETECTION ----------------
        results = plate_model(vehicle_img, verbose=False)

        plate_count = 0

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_crop = vehicle_img[y1:y2, x1:x2]

                if plate_crop.size == 0:
                    continue

                # Save plate crop in the same vehicle folder
                plate_filename = f"plate_{plate_count}.jpg"
                cv2.imwrite(os.path.join(vehicle_folder, plate_filename), plate_crop)
                plate_count += 1

        # print(f"Processed {vehicle_folder_name}: {plate_count} plates saved.")
