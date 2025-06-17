import os
import xml.etree.ElementTree as ET
import random
import shutil

# Paths
raw_data_path = r"C:\VBS\raw_data"
images_output = os.path.join(raw_data_path, "images")
labels_output = os.path.join(raw_data_path, "labels")

# Create output folders if not exist
for folder in ["train", "val", "test"]:
    os.makedirs(os.path.join(images_output, folder), exist_ok=True)
    os.makedirs(os.path.join(labels_output, folder), exist_ok=True)

# Classes list
CLASSES = ['apple', 'banana', 'carrot', 'corn', 'grapes',
           'kiwi', 'lettuce', 'onion', 'pineapple', 'potato', 'tomato']


def convert_bbox(size, box):
    """ Convert XML bounding box to YOLO format """
    dw = 1. / size[0]
    dh = 1. / size[1]
    xmin, xmax, ymin, ymax = box
    x = (xmin + xmax) / 2.0
    y = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def parse_xml(xml_path):
    """ Parse XML file to extract image size and object boxes """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    objects = []
    for obj in root.iter('object'):
        cls = obj.find('name').text.lower()
        if cls not in CLASSES:
            continue
        cls_id = CLASSES.index(cls)

        xmlbox = obj.find('bndbox')
        bbox = (int(xmlbox.find('xmin').text), int(xmlbox.find('xmax').text),
                int(xmlbox.find('ymin').text), int(xmlbox.find('ymax').text))

        bbox_yolo = convert_bbox((w, h), bbox)
        objects.append((cls_id, bbox_yolo))
    return objects, w, h


# Collect all XML and image files
all_xml_files = [f for f in os.listdir(raw_data_path) if f.endswith('.xml')]
all_images_files = [f for f in os.listdir(raw_data_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Make sure images have matching XMLs
xml_basenames = set([os.path.splitext(f)[0] for f in all_xml_files])
image_basenames = set([os.path.splitext(f)[0] for f in all_images_files])
common_files = list(xml_basenames.intersection(image_basenames))

print(f"Found {len(common_files)} valid image-annotation pairs.")

# Shuffle and split
random.seed(42)
random.shuffle(common_files)

n_total = len(common_files)
n_train = int(0.7 * n_total)
n_val = int(0.2 * n_total)
n_test = n_total - n_train - n_val

train_files = common_files[:n_train]
val_files = common_files[n_train:n_train + n_val]
test_files = common_files[n_train + n_val:]


def process_split(file_list, split_name):
    for base in file_list:
        xml_path = os.path.join(raw_data_path, base + ".xml")
        image_ext = None
        # find image extension
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = os.path.join(raw_data_path, base + ext)
            if os.path.exists(img_path):
                image_ext = ext
                break
        if image_ext is None:
            print(f"Image for {base} not found, skipping.")
            continue

        # Parse annotation
        objects, w, h = parse_xml(xml_path)
        if not objects:
            print(f"No valid objects found in {base}, skipping.")
            continue

        # Write label file
        label_path = os.path.join(labels_output, split_name, base + ".txt")
        with open(label_path, 'w') as f:
            for cls_id, (x, y, w_box, h_box) in objects:
                f.write(f"{cls_id} {x:.6f} {y:.6f} {w_box:.6f} {h_box:.6f}\n")

        # Copy image
        dst_image_path = os.path.join(images_output, split_name, base + image_ext)
        shutil.copy2(img_path, dst_image_path)


process_split(train_files, "train")
process_split(val_files, "val")
process_split(test_files, "test")

print("Dataset organized and split into train, val, and test.")
