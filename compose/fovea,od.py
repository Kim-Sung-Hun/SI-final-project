import os
import numpy as np

directory_path = '/data/sunghun/dataset/spare/HRF'

all_data = []
for file_name in os.listdir(directory_path):
    file_path = os.path.join(directory_path, file_name)
    data = np.load(file_path)
    all_data.append(data)

# numpy 배열로 변환
all_data = np.vstack(all_data)

# all_data의 첫번째 차원 값이 500 이하인 데이터만 선택
# all_data = all_data[all_data[:, 0] < 500]

# KDE 모델 설정 및 학습
kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(all_data)

# KDE 모델로부터 10개의 데이터 샘플 생성
new_samples = kde.sample(10)

# 선택적: 생성된 샘플과 원본 데이터를 시각화하지만, 화면에 표시하지 않고 파일로 저장
plt.scatter(all_data[:, 0], all_data[:, 1], c='blue', s=5, label='원본 데이터')
plt.scatter(new_samples[:, 0], new_samples[:, 1], c='red', marker='x', s=50, label='샘플링된 데이터')
plt.legend()
plt.title('KDE를 이용한 데이터 샘플링')
plt.xlabel('X')
plt.ylabel('Y')
# plt.show() 대신에 아래 코드 사용
plt.savefig('/data/sunghun/diffae_final/compose/sampled_data_plot.png')  # 파일 경로와 이름 설정
plt.close()  # 현재 그림을 닫음 (더 이상 사용하지 않을 때 권장)