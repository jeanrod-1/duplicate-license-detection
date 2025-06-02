import os
import random
import shutil
from PIL import Image, ImageDraw

def draw_yolo_bbox(image_path, label_path, save_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size

    if not os.path.exists(label_path):
        return  # skip if no label

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x_center, y_center, width, height = map(float, parts)
            xmin = (x_center - width / 2) * w
            ymin = (y_center - height / 2) * h
            xmax = (x_center + width / 2) * w
            ymax = (y_center + height / 2) * h
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)

    image.save(save_path)

def pick_and_draw_random_samples(image_dir, label_dir, output_dir, subset_name, n=2):
    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    random_samples = random.sample(images, min(n, len(images)))

    for img_file in random_samples:
        image_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, img_file.replace('.jpg', '.txt'))
        save_path = os.path.join(output_dir, f"{subset_name}_{img_file}")
        draw_yolo_bbox(image_path, label_path, save_path)

# Paths
base = "all_datasets/plates_from_cars"
output_dir = "new_dataset_temp_check"

os.makedirs(output_dir, exist_ok=True)

# Run for both train and val
pick_and_draw_random_samples(
    image_dir=os.path.join(base, "images/train"),
    label_dir=os.path.join(base, "labels/train"),
    output_dir=output_dir,
    subset_name="train"
)

pick_and_draw_random_samples(
    image_dir=os.path.join(base, "images/val"),
    label_dir=os.path.join(base, "labels/val"),
    output_dir=output_dir,
    subset_name="val"
)

print(f"✅ Imágenes guardadas en '{output_dir}' con bounding boxes.")
