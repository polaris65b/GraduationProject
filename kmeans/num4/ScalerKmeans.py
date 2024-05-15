import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 데이터 로딩
df = pd.read_csv('C:\\dev\\koreatech\\graduationProject\\kmeans\\efc\\재활훈련정보(최종).csv')

# 각 user_id에 대한 마지막 행을 선택합니다(가장 큰 train_id를 가진 행)
df = df.sort_values('train_id').groupby('user_id').last().reset_index()

# 사용할 열을 선택합니다
df = df[['height', 'age', 'BFP', 'grade', 'train_count']]

# 데이터 정규화
scaler = StandardScaler()
df[['height', 'age', 'BFP']] = scaler.fit_transform(df[['height', 'age', 'BFP']])

# 특성 리스트
features = ['height', 'age', 'BFP', 'train_count']

# K-평균 클러스터링 수행 및 그래프 생성
plt.figure(figsize=(18, 6))  # 전체 그림 크기 설정

for i, feature in enumerate(features, 1):  # enumerate 사용하여 인덱스와 함께 루프
    # K-평균 클러스터링 수행
    kmeans = KMeans(n_clusters=3, random_state=0)
    df['cluster'] = kmeans.fit_predict(df[[feature]])
    
    # subplot 추가
    plt.subplot(2, 2, i)  # 1행 3열의 i번째 위치에 subplot 생성
    plt.scatter(df[feature], df['grade'], c=df['cluster'], cmap='viridis')
    plt.xlabel(feature)
    plt.ylabel('Grade')
    plt.title(f'Grade vs. {feature}')
    plt.colorbar(label='Cluster ID')

plt.tight_layout()  # subplot 간격 자동 조정
plt.show()