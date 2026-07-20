# preprocess_floorplans.py
import cv2
import numpy as np
from PIL import Image
import pytesseract

def remove_text_from_floorplan(img_path, output_path):
    """Remove text annotations from floorplan"""
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect text regions using OCR
    boxes = pytesseract.image_to_boxes(gray)
    
    # Create mask for text regions
    mask = np.zeros(gray.shape, dtype=np.uint8)
    for b in boxes.splitlines():
        b = b.split()
        x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        cv2.rectangle(mask, (x, y), (w, h), 255, -1)
    
    # Inpaint (fill) text regions with surrounding pixels
    result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    
    cv2.imwrite(output_path, result)
    return output_path