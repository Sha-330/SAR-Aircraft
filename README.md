# SAR Aircraft Detection
## This project implements an aircraft detection system using the Ultralytics YOLOv8 model for Synthetic Aperture Radar (SAR) imagery. The model is trained to detect aircraft in SAR images, leveraging the YOLOv8 architecture for object detection.

### Requirements
```
Python 3.8 or higher
PyTorch (GPU support recommended for faster training)
Ultralytics YOLO library
Additional dependencies listed in requirements.txt
```
### Installation
```
git clone https://github.com/Sha-330/SAR-Aircraft
cd SAR-aircraft
```
```
Ensure you have PyTorch installed with CUDA support if you have a compatible GPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


Install Ultralytics:
pip install ultralytics
```


Dataset

```
Refer here
https://github.com/hust-rslab/SAR-aircraft-data
```

### Training
```
The training script (train.py) trains a YOLOv8 model for aircraft detection in SAR imagery. The script uses the following configuration:

Model: YOLOv8 (pretrained weights at last.pt or a configuration file like yolov8s.yaml)
Dataset: Specified in data.yaml
Epochs: 100
Image Size: 640x640 pixels
Batch Size: 16
Device: Automatically selects CUDA if available, otherwise CPU
Optimizer: SGD
Learning Rate: 0.0005 (with cosine learning rate scheduler)
Augmentation: Enabled
Patience: Early stopping after 10 epochs with no improvement
Output Directory: Results saved in runs/detect/sar_aircraft_detector

Run Training
Execute the training script:
python train.py

The script loads the model weights from:
C:.../SAR-aircraft/runs/detect/sar_aircraft_detector5/weights/last.pt

Ensure this file exists or replace it with the appropriate pretrained weights or model configuration (e.g., yolov8s.yaml).
Output
Training results, including the best model weights (best.pt), are saved in:
runs/detect/sar_aircraft_detector/

Usage

Inference:After training, use the trained model for inference on new SAR images:
from ultralytics import YOLO
model = YOLO('runs/detect/sar_aircraft_detector/weights/best.pt')
results = model.predict(source='path/to/image.jpg', save=True)


Evaluation:Evaluate the model on a validation set:
results = model.val()
print(f"Precision: {results.box.mp:.3f}")
print(f"Recall: {results.box.mr:.3f}")
print(f"mAP@0.5: {results.box.map50:.3f}")



Notes

Ensure the last.pt or best.pt weights file exists in the specified path before training or inference.
If you encounter a FileNotFoundError, verify the path to the weights file or dataset.
For GPU acceleration, ensure CUDA and cuDNN are properly installed and compatible with your PyTorch version.
Adjust hyperparameters (e.g., epochs, batch, lr0) based on your dataset size and requirements.

```

FileNotFoundError: Check the path to last.pt or data.yaml. Use double backslashes (\\) or raw strings (r'path\to\file') for Windows paths.
TypeError for Metrics: If accessing metrics like box.mp, use them as attributes (e.g., box.mp instead of box.mp()).
Out of Memory: Reduce batch size or use a smaller model (e.g., yolov8s.yaml instead of yolov8m.yaml).

License
This project is licensed under the MIT License.
