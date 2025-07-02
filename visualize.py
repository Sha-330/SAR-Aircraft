from ultralytics import YOLO
import cv2
import os

def main():
    model = YOLO('runs/detect/sar_aircraft_detector/weights/best.pt')

    test_dir = r'C:\Users\Ajmal\Downloads\Documents\program\SAR-aircraft\SAR-aircraft\yolo_dataset\images\test'
    out_dir = 'predictions'
    os.makedirs(out_dir, exist_ok=True)

    for img_name in os.listdir(test_dir):
        if not img_name.endswith('.bmp'):
            continue
        img_path = os.path.join(test_dir, img_name)
        results = model(img_path, conf=0.1, iou=0.7, save=False)
        result_img = results[0].plot()
        cv2.imwrite(os.path.join(out_dir, img_name), result_img)

    print("✅ Inference complete. Check 'predictions' folder.")

if __name__ == '__main__':
    main()
