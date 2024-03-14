# FACE ANTI-SPOOFING WITH YOLOV8 & FASTAPI

# DESCRIPTION
For this project, we are developing a face anti spoofing system with a pretrained yolov8 model. The trained model is later tested with FastAPI.
# DATASET
The dataset used is the *Large Crowdcollected Facial Anti-Spoofing Dataset*, a well knowend dataset used for face anti-spoofing model training. We used its training data large of 8299 images and divided it into train val and test with the following ratio, 80% 10% and 10%. The dataset has two classes: real and spoof. The fake or spoof faces are made from high quality records of the genuine/real faces.
# TRAINING PERFORMANCE
 - Confusion Matrix
![confusion_matrix](https://github.com/WENDGOUNDI/face_anti_spoofing_yolov8/assets/48753146/6e83bdc0-c563-4686-9f74-e3ec3f76452b)
 - Confusion Matrix Normalized
![confusion_matrix_normalized](https://github.com/WENDGOUNDI/face_anti_spoofing_yolov8/assets/48753146/d73a025a-a548-48ad-867c-01f14f7d56cc)
 - End Training Result
![results](https://github.com/WENDGOUNDI/face_anti_spoofing_yolov8/assets/48753146/a6c8c835-4355-461a-82cb-6c51b8092c9e)
 - Training Performance
![training_performance](https://github.com/WENDGOUNDI/face_anti_spoofing_yolov8/assets/48753146/8d486a49-c397-4f42-b340-659bd482b9f0)

# INFERENCE
We run the inference via FastAPI
 - Real face
![inference_real](https://github.com/WENDGOUNDI/face_anti_spoofing_yolov8/assets/48753146/f9b15127-ca6d-4b09-bffe-ec3c9bcfc45a)

 - Spoof face
![inference_spoof](https://github.com/WENDGOUNDI/face_anti_spoofing_yolov8/assets/48753146/96e15b3d-a811-4511-9f97-7b3b4072ee41)

# REFERENCE
Dataset link: https://www.kaggle.com/datasets/faber24/lcc-fasd
