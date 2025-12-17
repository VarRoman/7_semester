#%%
import albumentations as A
import cv2
import os
import glob
from tqdm import tqdm
from ultralytics import YOLO
import torch
#%%
base_dir = 'nike_dataset/train'
images_dir = os.path.join(base_dir, 'images')
labels_dir = os.path.join(base_dir, 'labels')

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomScale(scale_limit=0.1, p=0.5),
    A.Perspective(p=0.2),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(blur_limit=3, p=0.2),
    A.GaussNoise(p=0.2),
    A.HueSaturationValue(p=0.3),
    A.CoarseDropout(
        num_holes_range=(1, 8),
        hole_height_range=(8, 30),
        hole_width_range=(8, 30),
        p=0.2
    )
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def read_label(label_path):
    bboxes, classes = [], []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    classes.append(int(parts[0]))
                    bboxes.append([float(x) for x in parts[1:]])
    return bboxes, classes

def save_label(save_path, bboxes, classes):
    with open(save_path, 'w') as f:
        for cls, bbox in zip(classes, bboxes):
            bbox = [min(max(x, 0.0), 1.0) for x in bbox]
            f.write(f"{cls} {' '.join(map(str, bbox))}\n")

image_paths = glob.glob(os.path.join(images_dir, '*.jpg'))
print(f"Всього зображень: {len(image_paths)}")

LIMIT = 500
count = 0

for img_path in tqdm(image_paths, desc="Аугментація"):
    if count >= LIMIT:
        break

    if "_aug_" in img_path: continue

    image = cv2.imread(img_path)
    if image is None: continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    base_name = os.path.basename(img_path).rsplit('.', 1)[0]
    label_path = os.path.join(labels_dir, base_name + '.txt')

    bboxes, class_labels = read_label(label_path)

    if not bboxes: continue

    for i in range(1):
        try:
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)

            if len(augmented['bboxes']) > 0:
                new_filename = f"{base_name}_aug_{i}"

                out_img = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(images_dir, new_filename + '.jpg'), out_img)
                save_label(os.path.join(labels_dir, new_filename + '.txt'),
                           augmented['bboxes'],
                           augmented['class_labels'])
        except Exception:
            pass

    count += 1

print(f'Додано {count} нових зображень')
#%%
if torch.cuda.is_available():
    torch.cuda.empty_cache()

def train_nike():
    yaml_path = os.path.abspath('nike_dataset/data.yaml')

    model = YOLO('yolo11n.pt')
    results = model.train(
        data=yaml_path,
        epochs=3,
        imgsz=640,
        batch=32,
        device=0,
        name='nike_mk',
        patience=1,
        workers=0)

    print("Навчання завершено")

train_nike()
#%%
video_path = "nike_dataset/nike_video.webm"
model_path = "runs/detect/nike_mk3/weights/best.pt"

def process_video():
    model = YOLO(model_path)

    results = model.predict(
        source=video_path,
        save=True,
        conf=0.5,
        device=0,
        stream=True,
        project="runs/detect",
        name="video_result_nike")

    for result in results:
        pass
    print("Відео готове")

process_video()
#%%
