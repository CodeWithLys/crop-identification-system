import os
import xml.etree.ElementTree as ET

# Folder with images and XML files
raw_dir = r"C:\VBS\raw_data"


# Folder to save YOLO label txt files
labels_dir = 'raw_data_labels'
os.makedirs(labels_dir, exist_ok=True)

# Get sorted list of all classes (you can manually specify too)
classes = set()

# First pass: collect all class names
for file in os.listdir(raw_dir):
    if file.endswith('.xml'):
        xml_path = os.path.join(raw_dir, file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            cls = obj.find('name').text
            classes.add(cls)

# Sort classes and create list
classes = sorted(list(classes))
print("Classes found:", classes)

# Create a class to index dictionary
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

# Convert bounding box format to YOLO format
def convert_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x_center = (box[0] + box[1]) / 2.0 - 1
    y_center = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x_center *= dw
    w *= dw
    y_center *= dh
    h *= dh
    return (x_center, y_center, w, h)

# Process each XML file to create YOLO txt files
for file in os.listdir(raw_dir):
    if file.endswith('.xml'):
        xml_path = os.path.join(raw_dir, file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Image dimensions
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        label_lines = []

        for obj in root.findall('object'):
            cls = obj.find('name').text
            if cls not in class_to_idx:
                continue
            cls_id = class_to_idx[cls]

            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert_bbox((w, h), b)
            label_lines.append(f"{cls_id} " + " ".join([f"{a:.6f}" for a in bb]))

        # Save label file with same base filename as image
        label_filename = os.path.splitext(file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_filename)
        with open(label_path, 'w') as f:
            f.write("\n".join(label_lines))

print(f"YOLO label txt files created in '{labels_dir}' folder.")

import os
import random
import shutil

# Path to your raw_data folder where images and labels are stored
RAW_DATA_DIR = "raw_data"

# Assuming images are .jpg and labels are .txt with the same basename
images_dir = os.path.join(RAW_DATA_DIR, "images")
labels_dir = os.path.join(RAW_DATA_DIR, "labels")

# Where to save train and val splits
train_img_dir = os.path.join(RAW_DATA_DIR, "train", "images")
train_lbl_dir = os.path.join(RAW_DATA_DIR, "train", "labels")
val_img_dir = os.path.join(RAW_DATA_DIR, "val", "images")
val_lbl_dir = os.path.join(RAW_DATA_DIR, "val", "labels")

# Create directories if they don't exist
for folder in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
    os.makedirs(folder, exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg") or f.endswith(".png")]

# Shuffle for randomness
random.shuffle(image_files)

# Split ratio
split_ratio = 0.8
split_index = int(len(image_files) * split_ratio)

train_files = image_files[:split_index]
val_files = image_files[split_index:]


def move_files(file_list, img_dest, lbl_dest):
    for file_name in file_list:
        # Move image
        shutil.copy2(os.path.join(images_dir, file_name), os.path.join(img_dest, file_name))

        # Move label (.txt with same basename)
        label_file = os.path.splitext(file_name)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)

        if os.path.exists(label_path):
            shutil.copy2(label_path, os.path.join(lbl_dest, label_file))
        else:
            print(f"Warning: Label file {label_file} not found for image {file_name}")


# Move train files
move_files(train_files, train_img_dir, train_lbl_dir)

# Move val files
move_files(val_files, val_img_dir, val_lbl_dir)

print(f"Train set: {len(train_files)} images")
print(f"Validation set: {len(val_files)} images")
