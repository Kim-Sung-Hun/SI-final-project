import os
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
import numpy as np
from scipy.linalg import sqrtm
from PIL import Image

def get_features(directory, model, device, gt=False, type='fake'):
    
    if gt:
        
        transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    else:
        
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    features = []
    model.eval()
    
    for i, image_filename in enumerate(os.listdir(directory)):
        
        print(f'type : {type}, / {i+1} / {len(os.listdir(directory))}')
        
        image_path = os.path.join(directory, image_filename)
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(image)
            
        features.append(pred[0])
        
    features = torch.stack(features)
    
    return features

def calculate_fid(real_features, fake_features):
    
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    
    return fid

def main(real_dir, fake_dir):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = inception_v3(pretrained=True).to(device)
    model.fc = torch.nn.Identity()
    
    real_features = get_features(real_dir, model, device, gt=True, type='real').cpu().numpy()
    fake_features = get_features(fake_dir, model, device).cpu().numpy()
    
    fid_score = calculate_fid(real_features, fake_features)
    print(f'FID score: {fid_score}')

# 폴더 경로 지정 (예시 경로는 적절히 수정)
real_dir = '/DATA/sunghun/dataset/ReTree/test/images'
fake_dir = '/DATA/yohan/stylegan3/out'

main(real_dir, fake_dir)
