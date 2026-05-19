import os
import cv2
import numpy as np
from sklearn.cluster import KMeans

# ---------------- PATHS ----------------
input_root = "./outputs"
# output_root = "./output-color"
output_root = "./outputs"

valid_vehicle_prefix = ("car_", "truck_", "bus_")

os.makedirs(output_root, exist_ok=True)


# ---------------- SIMPLE COLOR NAME ----------------
def get_color_name(bgr):
    b, g, r = bgr

    if r > 150 and g < 80 and b < 80:
        return "Red"
    elif g > 150 and r < 80 and b < 80:
        return "Green"
    elif b > 150 and r < 80 and g < 80:
        return "Blue"
    elif r > 200 and g > 200 and b > 200:
        return "White"
    elif r < 50 and g < 50 and b < 50:
        return "Black"
    elif r > 150 and g > 150 and b < 80:
        return "Yellow"
    else:
        return "Unknown"


# ---------------- LOOP ----------------
for image_name in os.listdir(input_root):
    image_path = os.path.join(input_root, image_name)
    if not os.path.isdir(image_path):
        continue

    # Iterate over vehicle folders
    for vehicle_name in os.listdir(image_path):
        if not vehicle_name.startswith(valid_vehicle_prefix):
            continue

        vehicle_folder = os.path.join(image_path, vehicle_name)
        crop_path = os.path.join(vehicle_folder, "crop.jpg")

        if not os.path.exists(crop_path):
            print(f"crop.jpg not found in {vehicle_folder}")
            continue

        img = cv2.imread(crop_path)
        if img is None:
            print(f"Failed to read {crop_path}")
            continue

        # ---------------- COLOR DETECTION ----------------
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # remove dark & very bright areas
        lower_color = np.array([0, 40, 40])
        upper_color = np.array([180, 255, 255])
        mask_color = cv2.inRange(hsv, lower_color, upper_color)

        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 40, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        mask = cv2.bitwise_or(mask_color, mask_white)
        filtered = cv2.bitwise_and(img, img, mask=mask)

        pixels = filtered.reshape(-1, 3)
        pixels = pixels[np.any(pixels > 0, axis=1)]

        if len(pixels) == 0:
            continue

        # KMeans dominant color
        kmeans = KMeans(n_clusters=3, n_init=10)
        kmeans.fit(pixels)

        counts = np.bincount(kmeans.labels_)
        dominant_color = kmeans.cluster_centers_[np.argmax(counts)].astype(int)

        color_name = get_color_name(dominant_color)

        # ---------------- SAVE OUTPUT ----------------
        info_file = os.path.join(vehicle_folder, f"{vehicle_name}.txt")
        with open(info_file, "a") as f:
            f.write(f"Color: {color_name}\n")
            f.write(f"BGR: {dominant_color.tolist()}\n")

        # print(f"{vehicle_name} → {color_name} {dominant_color}")
