# import torch
# import cv2
# from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
# from torchvision.models.video import mc3_18, MC3_18_Weights

# weights = MC3_18_Weights.DEFAULT
# model = mc3_18(weights=weights)
# model.eval()

# cap = cv2.VideoCapture(0)

# # Prepare 16-frame clip (C, T, H, W)
# transform = Compose([
#     Resize((128, 171)),
#     CenterCrop(112),
#     ToTensor(),
#     Normalize(mean=[0.43216, 0.394666, 0.37645],
#               std=[0.22803, 0.22145, 0.216989])
# ])

# # Inference
# with torch.no_grad():
#     output = model(source = 0)  # (1, 101)
#     prediction = output.argmax(dim=1)

# print(f"Action: {ucf101_classes[prediction]}")

# cap.release()
# cv2.destroyAllWindows()

import sys
import os

# Get the absolute path of the directory containing the module
module_dir = os.path.abspath('human-action-classification/src')

# Insert it at the beginning of the system path
sys.path.insert(0, module_dir)

from hac import ActionPredictor

# Initialize with pose estimation
predictor = ActionPredictor(
    model_path=None,  # Uses pretrained ResNet50
    device='cuda',
    use_pose_estimation=True
)

# Predict from image
result = predictor.predict_image('training_data/handstand-1.jpg')

print(f"Pose: {result['pose']['class']}")
print(f"Action: {result['action']['top_class']}")
print(f"Confidence: {result['action']['top_confidence']:.2%}")