import os
import cv2
import torch
import torch.nn.functional as F
from torchvision.models import resnet18
from torch import nn
from torchvision import transforms
from PIL import Image

# ---------------- PATHS ----------------
input_root = "./outputs"  # from YOLO
# output_root = "./output-brand"  # new structured output
output_root = "./outputs"
valid_vehicle_prefix = ("car_", "truck_", "bus_")

os.makedirs(output_root, exist_ok=True)

# ---------------- LOAD MODEL ----------------
class_names = torch.load("models/car-brand/v1-1/classes.pth")

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
# model.load_state_dict(torch.load("car_brand.pth", map_location="cpu"))
model.load_state_dict(torch.load("models/car-brand/v1-1/car_brand.pth", map_location="cpu"))
model.eval()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # correct ImageNet mean
            std=[0.229, 0.224, 0.225],
        ),  # correct ImageNet std
    ]
)

# ---------------- LOOP ----------------
for image_name in os.listdir(input_root):
    image_path = os.path.join(input_root, image_name)
    if not os.path.isdir(image_path):
        continue

    # Loop over vehicle folders only
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

        # ---------------- BRAND PREDICTION ----------------
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        input_tensor = transform(img_pil).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        brand = class_names[pred.item()]
        confidence = conf.item() * 100

        # ---------------- SAVE TO info.txt ----------------
        info_file = os.path.join(vehicle_folder, f"{vehicle_name}.txt")
        with open(info_file, "a") as f:
            f.write(f"Brand: {brand}\n")
            f.write(f"Confidence: {confidence:.2f}%\n")

        # print(f"{vehicle_name} → {brand} ({confidence:.1f}%)")
