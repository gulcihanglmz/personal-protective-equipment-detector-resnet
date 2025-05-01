import cv2
import os

class_name = {
    11: 'hand',
    0: 'person',
    12: 'head',
    3: 'face',
    1: 'ear',
    7: 'tool',
    14: 'shoes',
    9: 'gloves',
    8: 'glasses',
    10: 'helmet',
    6: 'foot',
    5: 'face-mask-medical',
    16: 'safety-vest',
    2: 'ear-mufs',
    15: 'safety-suit',
    13: 'medical-suit',
    4: 'faceguard'
}

image_folder = r"C:\SH17-dataset\images"
label_folder = r"C:\SH17-dataset\labels"
output_folder = r"C:\SH17-dataset\output"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created: {output_folder}")

image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

count = 0
for image_filename in image_files:
    if count >= 30:
        break

    image_path = os.path.join(image_folder, image_filename)
    label_filename = os.path.splitext(image_filename)[0] + '.txt'
    label_path = os.path.join(label_folder, label_filename)

    print(f"Processing: {image_filename}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image: {image_path}")
        continue

    img_h, img_w, _ = image.shape

    if not os.path.exists(label_path):
        print(f"Warning: Label file not found: {label_path}. No boxes will be drawn for this image.")
        output_path = os.path.join(output_folder, f"vis_{image_filename}")
        cv2.imwrite(output_path, image)
        print(f"Saved (no labels): {output_path}")
        count += 1
        continue

    try:
        with open(label_path, 'r') as f:
            labels = f.readlines()
    except Exception as e:
        print(f"Error: Could not read label file: {label_path} - {e}")
        continue

    for label in labels:
        try:
            parts = label.strip().split()
            if len(parts) != 5:
                print(f"Warning: Invalid label format ({label_path}): {label.strip()}")
                continue

            class_id = int(parts[0])
            x_center_norm = float(parts[1])
            y_center_norm = float(parts[2])
            width_norm = float(parts[3])
            height_norm = float(parts[4])

            x_center = x_center_norm * img_w
            y_center = y_center_norm * img_h
            box_w = width_norm * img_w
            box_h = height_norm * img_h

            x1 = int(x_center - box_w / 2)
            y1 = int(y_center - box_h / 2)
            x2 = int(x_center + box_w / 2)
            y2 = int(y_center + box_h / 2)

            label_name = class_name.get(class_id, f'ID_{class_id}')  # Show ID_X for unknown IDs

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            text = f"{label_name}"
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 1)
            cv2.rectangle(image, (x1, y1 - text_height - baseline), (x1 + text_width, y1), (0, 255, 0), -1)
            cv2.putText(image, text, (x1, y1 - baseline // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 1, cv2.LINE_AA)

        except ValueError as ve:
            print(f"Warning: Invalid value ({label_path}): {label.strip()} - {ve}")
        except Exception as e:
            print(f"Error: An error occurred while processing the label ({label_path}): {label.strip()} - {e}")

    output_path = os.path.join(output_folder, f"vis_{image_filename}")
    try:
        cv2.imwrite(output_path, image)
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Error: Could not save image: {output_path} - {e}")

    count += 1

print(f"\nProcessing complete. {count} images visualized and saved to '{output_folder}'.")
