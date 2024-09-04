import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet101
from PIL import Image
from tqdm import tqdm
import numpy as np


def extract_features(image_root, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = resnet101(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-2])
    model.eval()
    model.cuda()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_paths = [os.path.join(image_root, img) for img in os.listdir(image_root) if img.endswith('.jpg')]

    for image_path in tqdm(image_paths):
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).cuda()

        with torch.no_grad():
            features = model(image).squeeze().cpu().numpy()

        output_path = os.path.join(output_dir, os.path.basename(image_path).replace('.jpg', '.npy'))
        np.save(output_path, features)

# Extract features for training and validation images
extract_features('/mnt/c/Users/muham/Downloads/coco/train2014', '/mnt/c/Users/muham/Downloads/coco/feats/train2014')
extract_features('/mnt/c/Users/muham/Downloads/coco/val2014', '/mnt/c/Users/muham/Downloads/coco/feats/val2014')
