import numpy as np

# 파일 경로
fovea_path = '/data/sunghun/dataset/full_data/train/fovea/90_image.npy'
od_path = '/data/sunghun/dataset/full_data/train/od/90_image.npy'

# 파일 불러오기
fovea_coords = np.load(fovea_path)
od_coords = np.load(od_path)

# 좌표 확인
print("Fovea 좌표 (x, y):", fovea_coords)
print("OD 좌표 (x, y):", od_coords)