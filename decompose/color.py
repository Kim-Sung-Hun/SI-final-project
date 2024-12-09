import numpy as np
import matplotlib.pyplot as plt

# 파일 경로
file_path = '/data/sunghun/dataset/full_data/train/color_thief/134_image.npy'

# 파일 불러오기
flattened_colors = np.load(file_path)

# (24,) 형식을 (8,3) 형식으로 변환
colors = flattened_colors.reshape((8, 3))

# 팔레트 저장
def save_color_palette(colors, save_path):
    fig, ax = plt.subplots(1, 8, figsize=(12, 2), subplot_kw=dict(xticks=[], yticks=[], frame_on=False))
    
    for i, color in enumerate(colors):
        ax[i].imshow([[color / 255]])  # 색상을 0-1 범위로 변환하여 표시

    plt.savefig(save_path, bbox_inches='tight')  # 파일로 저장
    plt.close(fig)  # 저장 후 창을 닫음

save_path = '/data/sunghun/dataset/full_data/train/color_palette.png'
save_color_palette(colors, save_path)