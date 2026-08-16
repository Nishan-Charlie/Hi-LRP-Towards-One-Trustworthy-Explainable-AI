import os
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET

def setup_demo_data():
    base_dir = "data/demo"
    img_dir = os.path.join(base_dir, "images")
    bbox_dir = os.path.join(base_dir, "bboxes")
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(bbox_dir, exist_ok=True)
    
    # 1. Create a synthetic image (e.g. a bird)
    img = Image.new('RGB', (224, 224), color = (73, 109, 137))
    d = ImageDraw.Draw(img)
    # Draw a yellow box as a fake object
    d.rectangle([(50, 50), (150, 150)], fill=(255, 255, 0))
    img.save(os.path.join(img_dir, 'synthetic_01.jpg'))
    
    # 2. Create the bounding box XML
    root = ET.Element("annotation")
    obj = ET.SubElement(root, "object")
    bndbox = ET.SubElement(obj, "bndbox")
    ET.SubElement(bndbox, "xmin").text = "50"
    ET.SubElement(bndbox, "ymin").text = "50"
    ET.SubElement(bndbox, "xmax").text = "150"
    ET.SubElement(bndbox, "ymax").text = "150"
    
    tree = ET.ElementTree(root)
    tree.write(os.path.join(bbox_dir, "synthetic_01.xml"))
    
    # 3. Create labels file (class 11 is goldfinch in imagenet)
    with open(os.path.join(base_dir, "labels.txt"), "w") as f:
        f.write("synthetic_01.jpg 11\n")

    print("Demo data generated.")

if __name__ == "__main__":
    setup_demo_data()
