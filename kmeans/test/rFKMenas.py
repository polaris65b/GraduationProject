import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# 데이터 로딩
df = pd.read_csv('C:\\dev\\koreatech\\graduationProject\\MRT_koreatech\\kmeans\\재활훈련정보_최종.csv')

# 각 user_id에 대한 마지막 행을 선택
df = df.sort_values('train_id').groupby('user_id', as_index=False).last()

# 사용할 열을 선택합니다
df = df[['user_id', 'height', 'age', 'BFP', 'grade', 'train_count']]

# 특성 리스트
features = ['height', 'age', 'BFP', 'train_count']

# 환자의 user_id 입력
user_id = int(input("환자의 user_id를 입력하세요: "))

# 해당 환자의 데이터 찾기
patient_data = df[df['user_id'] == user_id]

plt.figure(figsize=(18, 12))  # 전체 그림 크기 설정

for i, feature in enumerate(features, 1):
    # K-평균 클러스터링 수행
    kmeans = KMeans(n_clusters=3, random_state=0)
    df['cluster'] = kmeans.fit_predict(df[[feature]])
    
    plt.subplot(2, 2, i)
    plt.scatter(df[feature], df['grade'], c=df['cluster'], cmap='viridis')
    plt.xlabel(feature)
    plt.ylabel('Grade')
    plt.title(f'Grade vs. {feature}')
    plt.colorbar(label='Cluster ID')

    # 입력된 user_id의 군집 ID 찾기
    user_cluster_id = df[df['user_id'] == user_id]['cluster'].iloc[0]
    # 동일한 군집에 속한 데이터의 grade 값들을 추출
    cluster_grades = df[df['cluster'] == user_cluster_id]['grade']
    # 입력된 user_id의 grade 값
    user_grade = patient_data['grade'].iloc[0]
    # 상위 몇 %에 해당하는지 계산
    percentile = (np.sum(cluster_grades <= user_grade) / len(cluster_grades)) * 100
    
    plt.scatter(patient_data[feature], patient_data['grade'], c='red', s=200, marker='*', edgecolors='black', 
                label=f'User ID {user_id} (Top {percentile:.2f}%) in Cluster')
    plt.legend()
    
plt.tight_layout()  # subplot 간격 자동 조정
plt.show()