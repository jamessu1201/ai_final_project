#!/usr/bin/env python3
"""
prepare_control_images.py
為 ControlNet 實驗準備 control images
Person C - Day 2 使用
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import urllib.request

def create_simple_shapes(output_dir="control_images/simple_shapes"):
    """建立簡單的幾何圖形作為 control"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(" Creating simple geometric shapes...")
    
    shapes = []
    
    # 1. 圓形
    print("  Creating circle...")
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.circle(img, (256, 256), 150, (255, 255, 255), -1)
    Image.fromarray(img).save(f"{output_dir}/circle.png")
    shapes.append("circle")
    
    # 2. 方形
    print("  Creating square...")
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.rectangle(img, (150, 150), (350, 350), (255, 255, 255), -1)
    Image.fromarray(img).save(f"{output_dir}/square.png")
    shapes.append("square")
    
    # 3. 三角形
    print("  Creating triangle...")
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    pts = np.array([[256, 100], [100, 400], [400, 400]], np.int32)
    cv2.fillPoly(img, [pts], (255, 255, 255))
    Image.fromarray(img).save(f"{output_dir}/triangle.png")
    shapes.append("triangle")
    
    # 4. 星形
    print("  Creating star...")
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    pts = []
    for i in range(10):
        angle = i * 36 * np.pi / 180
        r = 150 if i % 2 == 0 else 60
        x = int(256 + r * np.cos(angle - np.pi/2))
        y = int(256 + r * np.sin(angle - np.pi/2))
        pts.append([x, y])
    pts = np.array(pts, np.int32)
    cv2.fillPoly(img, [pts], (255, 255, 255))
    Image.fromarray(img).save(f"{output_dir}/star.png")
    shapes.append("star")
    
    # 5. 簡單的房子輪廓
    print("  Creating house...")
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    # 房子主體
    cv2.rectangle(img, (150, 250), (362, 450), (255, 255, 255), -1)
    # 屋頂
    roof_pts = np.array([[256, 150], [100, 250], [412, 250]], np.int32)
    cv2.fillPoly(img, [roof_pts], (255, 255, 255))
    # 門
    cv2.rectangle(img, (220, 350), (292, 450), (0, 0, 0), -1)
    # 窗戶
    cv2.rectangle(img, (170, 280), (220, 330), (0, 0, 0), -1)
    cv2.rectangle(img, (292, 280), (342, 330), (0, 0, 0), -1)
    Image.fromarray(img).save(f"{output_dir}/house.png")
    shapes.append("house")
    
    print(f" Created {len(shapes)} simple shapes in {output_dir}/")
    return shapes

def download_reference_images(output_dir="control_images/references"):
    """從 Unsplash 下載一些參考圖片"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n Downloading reference images from Unsplash...")
    
    # 高質量的測試圖片 URLs (512x512)
    test_images = {
        "cat": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=512&h=512&fit=crop",
        "building": "https://images.unsplash.com/photo-1520763185298-1b434c919102?w=512&h=512&fit=crop",
        "mountain": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=512&h=512&fit=crop",
        "portrait": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=512&h=512&fit=crop",
        "cityscape": "https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b?w=512&h=512&fit=crop",
    }
    
    downloaded = []
    for name, url in test_images.items():
        output_path = Path(output_dir) / f"{name}.jpg"
        try:
            print(f"  Downloading {name}...")
            urllib.request.urlretrieve(url, output_path)
            downloaded.append(name)
        except Exception as e:
            print(f"    Failed to download {name}: {e}")
    
    print(f" Downloaded {len(downloaded)} reference images in {output_dir}/")
    return downloaded

def preview_images(shapes_dir="control_images/simple_shapes", 
                   refs_dir="control_images/references"):
    """顯示已準備的圖片統計"""
    print("\n" + "="*60)
    print(" Control Images Summary")
    print("="*60)
    
    shapes_path = Path(shapes_dir)
    refs_path = Path(refs_dir)
    
    if shapes_path.exists():
        shapes = list(shapes_path.glob("*.png"))
        print(f"\n Simple Shapes: {len(shapes)} images")
        for img in shapes:
            print(f"  - {img.name}")
    
    if refs_path.exists():
        refs = list(refs_path.glob("*.jpg"))
        print(f"\n Reference Images: {len(refs)} images")
        for img in refs:
            print(f"  - {img.name}")
    
    total = len(list(shapes_path.glob("*.png"))) + len(list(refs_path.glob("*.jpg")))
    print(f"\n Total control images prepared: {total}")
    print("="*60)

def main():
    print(" Control Images Preparation")
    print("Person C - Gen Comparison Project")
    print("="*60)
    print()
    
    # 選項
    print("Choose preparation method:")
    print("  1. Create simple geometric shapes (FAST, ~5 seconds)")
    print("  2. Download reference images from Unsplash (SLOW, ~30 seconds)")
    print("  3. Both (RECOMMENDED)")
    print()
    
    choice = input("Enter your choice (1/2/3) [default: 3]: ").strip() or "3"
    print()
    
    if choice in ["1", "3"]:
        create_simple_shapes()
    
    if choice in ["2", "3"]:
        download_reference_images()
    
    # 顯示統計
    preview_images()
    
    print("\n Next step:")
    print("  Run: python person_c_controlnet.py")
    print("       (to start ControlNet experiments)")

if __name__ == "__main__":
    main()