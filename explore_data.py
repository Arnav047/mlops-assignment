# Explore_data.py
import os
import numpy as np
from pathlib import Path
from PIL import Image

def analyze_dataset():
    data_dir = Path("./data/PennFudanPed")
    img_dir = data_dir / "PNGImages"
    mask_dir = data_dir / "PedMasks"

    images = sorted(os.listdir(img_dir))
    masks = sorted(os.listdir(mask_dir))

    total_images = len(images)
    total_objects = 0
    box_widths = []
    box_heights = []

    print("--- Penn-Fudan Pedestrian Dataset Analysis ---")
    print(f"Total number of images: {total_images}")

    # Loop through all masks to calculate bounding box stats
    for mask_name in masks:
        mask_path = mask_dir / mask_name
        mask = Image.open(mask_path)
        mask_np = np.array(mask)

        # Unique pixel values correspond to instance IDs (0 is background)
        obj_ids = np.unique(mask_np)[1:]
        total_objects += len(obj_ids)

        # Get bounding boxes for sizing analysis
        for obj_id in obj_ids:
            pos = np.where(mask_np == obj_id)
            xmin, xmax = np.min(pos[1]), np.max(pos[1])
            ymin, ymax = np.min(pos[0]), np.max(pos[0])
            
            box_widths.append(xmax - xmin)
            box_heights.append(ymax - ymin)

    # Compute descriptive metrics
    avg_boxes_per_img = total_objects / total_images
    avg_width = np.mean(box_widths)
    avg_height = np.mean(box_heights)

    print(f"Total pedestrian instances found: {total_objects}")
    print(f"Average pedestrians per image: {avg_boxes_per_img:.2f}")
    print(f"Average bounding box dimensions: {avg_width:.1f}x{avg_height:.1f} pixels (WxH)")
    print(f"Typical aspect ratio (H/W): {avg_height / avg_width:.2f} (Vertically elongated, typical for standing people)")
    print("---------------------------------------------")

if __name__ == "__main__":
    analyze_dataset()