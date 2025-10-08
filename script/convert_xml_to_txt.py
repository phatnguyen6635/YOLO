import os
import xml.etree.ElementTree as ET
import argparse
import numpy as np

# Định nghĩa label_map để ánh xạ tên lớp sang ID lớp của YOLO
label_map = {
    'sample': 0,
}

def parse_labelme_xml(xml_path):
    """Đọc file XML LabelMe và trích xuất thông tin bounding box."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Lấy kích thước ảnh
        image_size = root.find('imagesize')
        img_width = float(image_size.find('ncols').text)
        img_height = float(image_size.find('nrows').text)
        
        bboxes = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in label_map:
                print(f"Cảnh báo: Lớp {name} không có trong label_map, bỏ qua.")
                continue
            
            class_id = label_map[name]
            polygon = obj.find('polygon')
            pts = polygon.findall('pt')
            if len(pts) != 4:
                print(f"Cảnh báo: Polygon trong {xml_path} có {len(pts)} điểm, cần 4 điểm.")
                continue
            
            # Lấy tọa độ bounding box (giả sử theo thứ tự top-left, top-right, bottom-right, bottom-left)
            x1 = float(pts[0].find('x').text)  # Top-left x
            y1 = float(pts[0].find('y').text)  # Top-left y
            x2 = float(pts[2].find('x').text)  # Bottom-right x
            y2 = float(pts[2].find('y').text)  # Bottom-right y
            
            bboxes.append({
                'class_id': class_id,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            })
        
        return bboxes, img_width, img_height
    except Exception as e:
        print(f"Lỗi khi đọc file XML {xml_path}: {e}")
        return None, None, None

def convert_to_yolo_format(bboxes, img_width, img_height):
    """Chuyển đổi tọa độ bounding box sang định dạng YOLO."""
    yolo_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        class_id = bbox['class_id']
        
        # Tính toán tọa độ YOLO
        x_center = (x1 + x2) / (2 * img_width)
        y_center = (y1 + y2) / (2 * img_height)
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        # Đảm bảo giá trị nằm trong khoảng [0, 1]
        x_center = np.clip(x_center, 0, 1)
        y_center = np.clip(y_center, 0, 1)
        width = np.clip(width, 0, 1)
        height = np.clip(height, 0, 1)
        
        yolo_bboxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_bboxes

def save_yolo_txt(yolo_bboxes, output_path):
    """Lưu danh sách bounding box YOLO vào file .txt."""
    with open(output_path, 'w') as f:
        for bbox in yolo_bboxes:
            f.write(bbox + '\n')

def main(input_folder, output_folder):
    """Chuyển đổi tất cả file XML trong thư mục đầu vào sang định dạng YOLO."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.xml'):
            xml_path = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]
            output_txt_path = os.path.join(output_folder, base_name + '.txt')
            
            # Đọc và phân tích file XML
            bboxes, img_width, img_height = parse_labelme_xml(xml_path)
            if bboxes is None:
                continue
            
            # Chuyển đổi sang định dạng YOLO
            yolo_bboxes = convert_to_yolo_format(bboxes, img_width, img_height)
            
            # Lưu file .txt
            save_yolo_txt(yolo_bboxes, output_txt_path)
            print(f"Đã chuyển đổi {xml_path} sang {output_txt_path}")

if __name__ == "__main__":
    input_folder = '/home/phatnguyen/Documents/YOLO/data/augumation/val/'
    output_folder = '/home/phatnguyen/Documents/YOLO/data/augumation/val/label/'
    main(input_folder, output_folder)