import os
import cv2
import xml.etree.ElementTree as ET
import argparse
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from tqdm import tqdm

# Định nghĩa label_map và class_map (có thể chỉnh sửa theo nhu cầu)
label_map = {
    'sample': 0
}

class_map = {
    'sample': (0, 0, 255)
}

def get_bounding_boxes(tree):
    """Trích xuất danh sách bounding box từ file XML."""
    bboxes = []
    for obj in tree.findall('object'):
        polygon = obj.find('polygon')
        pts = polygon.findall('pt')
        if len(pts) == 4:
            x1 = float(pts[0].find('x').text)  # Top-left
            y1 = float(pts[0].find('y').text)
            x2 = float(pts[2].find('x').text)  # Bottom-right
            y2 = float(pts[2].find('y').text)
            name = obj.find('name').text
            bb = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=name)
            bboxes.append(bb)
    return bboxes

def update_xml_tree(tree, augmented_bboxes, new_filename):
    """Cập nhật cây XML với tọa độ bounding box mới và tên file mới."""
    tree.find('filename').text = new_filename
    objects = tree.findall('object')
    for obj, aug_bb in zip(objects, augmented_bboxes):
        polygon = obj.find('polygon')
        pts = polygon.findall('pt')
        # Cập nhật tọa độ cho 4 điểm
        pts[0].find('x').text = str(aug_bb.x1)  # Top-left
        pts[0].find('y').text = str(aug_bb.y1)
        pts[1].find('x').text = str(aug_bb.x2)  # Top-right
        pts[1].find('y').text = str(aug_bb.y1)
        pts[2].find('x').text = str(aug_bb.x2)  # Bottom-right
        pts[2].find('y').text = str(aug_bb.y2)
        pts[3].find('x').text = str(aug_bb.x1)  # Bottom-left
        pts[3].find('y').text = str(aug_bb.y2)
    return tree

def main(input_folder, output_folder, n):
    """Hàm chính để thực hiện augmentation và cập nhật file XML."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_extensions = ['.jpg', '.jpeg', '.png']

    for filename in tqdm(os.listdir(input_folder)):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_path = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]
            xml_path = os.path.join(input_folder, base_name + '.xml')

            if not os.path.exists(xml_path):
                print(f"Không tìm thấy file XML cho {filename}")
                continue

            # Đọc ảnh
            image = cv2.imread(image_path)
            if image is None:
                print(f"Không thể tải ảnh {filename}")
                continue

            # Parse XML
            tree = ET.parse(xml_path)
            bboxes = get_bounding_boxes(tree)

            # Thực hiện n lần augmentation
            for i in range(1, n + 1):
                # Định nghĩa chuỗi augmentation
                seq = iaa.Sequential([
                    iaa.Fliplr(0.5),  # Lật ngang 50% trường hợp
                    iaa.Flipud(0.5),  # Lật dọc 50% trường hợp
                    iaa.Affine(rotate=(-3, 3)),
                    iaa.AdditiveGaussianNoise(scale=(0, 0.01*255)),  # Add noise
                    iaa.Affine(translate_px={"x": (-5, 5), "y": (-5, 5)}, mode="constant", cval=0),              
		            iaa.Multiply((0.95, 1.05))
                ])

                # Áp dụng augmentation
                bbs = BoundingBoxesOnImage(bboxes, shape=image.shape)
                image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

                # Lưu ảnh đã augmentation
                aug_filename = f"{base_name}_aug_{i}.jpg"
                aug_image_path = os.path.join(output_folder, aug_filename)
                cv2.imwrite(aug_image_path, image_aug)

                # Cập nhật và lưu file XML
                aug_tree = update_xml_tree(tree, bbs_aug.bounding_boxes, aug_filename)
                aug_xml_filename = f"{base_name}_aug_{i}.xml"
                aug_xml_path = os.path.join(output_folder, aug_xml_filename)
                aug_tree.write(aug_xml_path)

                # Tạo và lưu ảnh debug
                debug_image = image_aug.copy()
                for bb in bbs_aug.bounding_boxes:
                    name = bb.label
                    color = class_map.get(name, (0, 255, 0))  # Màu mặc định là xanh lá
                    x1, y1, x2, y2 = int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)
                    cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, 2)
                    if name in label_map:
                        label_num = label_map[name]
                        text = str(label_num)
                        text_pos = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 20)
                        cv2.putText(debug_image, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                debug_filename = f"{base_name}_aug_{i}_debug.jpg"
                debug_path = os.path.join(output_folder, debug_filename)
                cv2.imwrite(debug_path, debug_image)
                
if __name__ == "__main__":
    input_folder = '/home/phatnguyen/Documents/YOLO/data/backup/'
    output_folder = '/home/phatnguyen/Documents/YOLO/data/augumation/val/'
    n = 2
    main(input_folder, output_folder, n)
