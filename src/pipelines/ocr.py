import os
import cv2
import matplotlib.pyplot as plt

# Fix Paddle MKLDNN issues (Windows)
os.environ["FLAGS_use_mkldnn"] = "0"

from paddleocr import PaddleOCR

# ---------------- INIT OCR ----------------
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='ar',   # change to 'en' if needed
    show_log=False
)

# ---------------- PATHS ----------------
input_root = "./outputs"
output_root = "./outputs"
# output_root = "./output-plate-detection"

os.makedirs(output_root, exist_ok=True)

valid_ext = (".jpg", ".jpeg", ".png", ".bmp")

# ---------------- PROCESS ALL IMAGES ----------------
for img_name in os.listdir(input_root):

    img_folder = os.path.join(input_root, img_name)
    # print("This is screen folder: ", img_folder)


    if not os.path.isdir(img_folder):
        continue

    for car_name in os.listdir(img_folder):

        car_folder = os.path.join(img_folder, car_name)
        # print("This is car folder: ", car_folder)

        if not os.path.isdir(car_folder):
            continue

        # Look for car image inside folder
        car_image_path = None

        for f in os.listdir(car_folder):
            if f.lower().endswith(valid_ext) and "crop" in f.lower():
                car_image_path = os.path.join(car_folder, f)
                break

        # fallback: take first image if naming differs
        if car_image_path is None:
            for f in os.listdir(car_folder):
                if f.lower().endswith(valid_ext):
                    car_image_path = os.path.join(car_folder, f)
                    break

        if car_image_path is None:
            print(f"⚠ No image found in {car_folder}")
            continue

        img = cv2.imread(car_image_path)

        if img is None:
            print(f"⚠ Cannot read {car_image_path}")
            continue

        # ---------------- OCR ----------------
        result = ocr.ocr(img)

        plate_texts = []
        confidences = []

        if result and result[0]:

            for line in result[0]:

                if line is None or len(line) < 2:
                    continue

                text = line[1][0] if line[1] else ""
                conf = line[1][1] if line[1] else 0

                if text:
                    plate_texts.append(text)
                    confidences.append(conf)

            plate_text = " ".join(plate_texts).strip()

            if len(confidences) > 0:
                avg_conf = sum(confidences) / len(confidences)
            else:
                avg_conf = 0

        else:
            plate_text = "No_Plate_Detected"
            avg_conf = 0

        # ---------------- SAFE FILE NAME ----------------
        safe_text = plate_text.replace(" ", "_").replace("/", "_")

        # ---------------- OUTPUT DIR ----------------
        out_dir = os.path.join(output_root, img_name, car_name)
        os.makedirs(out_dir, exist_ok=True)

        # Save image
        # cv2.imwrite(os.path.join(out_dir, "car_image.jpg"), img)

        # Save plate text as filename (requested)
        text_file_path = os.path.join(out_dir, f"{car_name}.txt")
        with open(text_file_path, "a", encoding="utf-8") as f:
            f.write(f"Plate Text: {plate_text}\n")
            f.write(f"Confidence: {avg_conf:.2f}\n")

        # ---------------- DISPLAY ----------------
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # print(f"\n📌 {img_name}/{car_name}")
        # print(f"🔤 Plate: {plate_text} | Conf: {avg_conf:.2f}")