from ultralytics import YOLO
import cv2
import os

# ---------------- LOAD MODEL ----------------
model = YOLO("./models/yolov8m.pt")

# ---------------- INPUT / OUTPUT ----------------
input_folder = "./input_image"
output_root = "./outputs"

os.makedirs(output_root, exist_ok=True)
valid_ext = (".jpg", ".jpeg", ".png", ".bmp")

# Get list of image files
images = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_ext)]

if len(images) == 0:
    print(
        "📂 No images found in input folder. Please add images to 'input_image' folder."
    )

# ---------------- LOOP THROUGH IMAGES ----------------
for image_name in images:

    image_path = os.path.join(input_folder, image_name)
    img = cv2.imread(image_path)

    if img is None:
        print(f"Skipping: {image_name}")
        continue

    base_name = os.path.splitext(image_name)[0]
    base_output_dir = os.path.join(output_root, base_name)
    os.makedirs(base_output_dir, exist_ok=True)

    # Save a copy of the original input image
    # cv2.imwrite(os.path.join(base_output_dir, "input_image.jpg"), img)

    # ---------------- DETECTION ----------------
    results = model(img, verbose=False)
    output_img = img.copy()
    count = 0

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]

            if label not in ["car", "truck", "bus"]:
                continue

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # ---------------- VEHICLE FOLDER ----------------
            vehicle_folder_name = f"{label}_{count}"
            vehicle_folder = os.path.join(base_output_dir, vehicle_folder_name)
            os.makedirs(vehicle_folder, exist_ok=True)

            # Save crop.jpg
            cv2.imwrite(os.path.join(vehicle_folder, "crop.jpg"), crop)

            # Save info.txt placeholder for later steps
            info_path = os.path.join(vehicle_folder, f"{vehicle_folder_name}.txt")
            with open(info_path, "a") as f:
                f.write(f"Label: {label}\nConfidence: {conf:.2f}\n")

            # Draw detection on output image
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                output_img,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            count += 1

    # Save final output image with detections
    cv2.imwrite(os.path.join(base_output_dir, "output_image.jpg"), output_img)
