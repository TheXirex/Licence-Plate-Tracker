# Licence-Plate-Tracker
## Context
License plate tracking plays an important role in various areas, from road safety and transportation management to improving business processes. It enables easy monitoring of vehicle movements, provides quick response to emergency situations, and can also be used to improve logistics and optimize costs in logistics companies.

**Main goal:** 
- get a dataset to train the model; 
- train the selected model from scratch; 
- plug in algorithms for tracking and license plate text reading.

## Selected Model and Dataset
I took a dataset from [Roboflow](https://universe.roboflow.com/augmented-startups/vehicle-registration-plates-trudk/dataset/2) for training the YOLOv8n model.

## Results
https://github.com/TheXirex/Licence-Plate-Tracker/assets/104722568/1a9e3a52-2fd3-4f5e-bdb4-5ad39fa6c3bc

## Requirements
1. numpy==1.26.1
2. opencv-python==4.8.1
3. ultralytics==8.0.206
4. deep_sort_realtime==1.3.2
5. easyOCR==1.7.1
6. roboflow==1.1.9