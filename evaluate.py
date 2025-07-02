from ultralytics import YOLO

def main():
    model = YOLO(r'C:\Users\Ajmal\Downloads\Documents\program\SAR-aircraft\runs\detect\sar_aircraft_detector5\weights\best.pt')
    metrics = model.val(data='data.yaml', split='test', conf=0.1, iou=0.7)

    # Access metrics from metrics.box (the real Metric object)
    box = metrics.box

    print("\n📊 Evaluation Metrics:")
    print(f"Precision (mean): {box.mp():.3f}")
    print(f"Recall (mean):    {box.mr():.3f}")
    print(f"mAP@0.5:          {box.map50():.3f}")
    print(f"mAP@0.5:0.95:     {box.map():.3f}")

if __name__ == '__main__':
    main()
