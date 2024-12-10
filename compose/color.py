import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# 데이터 로드
data_path = '/data/sunghun/dataset/full_data/test/color_thief/combined_colors.npy'
rgb_colors = np.load(data_path)

# KDE 모델 생성 및 학습
bandwidth = 0.5
kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
kde.fit(rgb_colors)

# 샘플링
num_samples = 8
samples = kde.sample(num_samples)

# 샘플 값을 0-255 범위로 변환
samples = np.clip(samples, 0, 255).astype(int)

# 팔레트 시각화 및 저장 함수
def save_palette(colors, filename):
    n = len(colors)
    fig, ax = plt.subplots(1, n, figsize=(n, 1),
                           subplot_kw=dict(xticks=[], yticks=[], frame_on=False))
    for i, color in enumerate(colors):
        ax[i].imshow([[color / 255.0]])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

# 저장 경로 지정
save_path = '/data/sunghun/diffae_final/compose/color/palette.png'

# 팔레트 시각화 및 저장
save_palette(samples, save_path)
