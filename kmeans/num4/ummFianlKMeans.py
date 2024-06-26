import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 데이터 로딩
#df = pd.read_csv('D:\\dev\\GraduationProject\\kmeans\\efc\\재활훈련정보(최종).csv')
df = pd.read_csv('C:\\dev\\koreatech\\graduationProject\\etc\\재활훈련정보_최종.csv',encoding='cp949')

# 각 user_id에 대한 마지막 행을 선택합니다(가장 큰 train_id를 가진 행)
df = df.sort_values('train_id').groupby('user_id').last().reset_index()

# 사용할 열을 선택합니다
df = df[['height', 'age', 'BFP', 'grade', 'train_count']]

# 데이터 정규화 (train_count 포함)
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[['height', 'age', 'BFP', 'train_count']] = scaler.fit_transform(df_scaled[['height', 'age', 'BFP', 'train_count']])

# 가중치 설정
weights = {'height': 0.3, 'age': 0.18, 'BFP': 0.15, 'train_count': 0.37}

# 가중치 적용하여 'total' 특성 생성 (정규화된 train_count 사용)
df_scaled['total'] = df_scaled['height'] * weights['height'] + df_scaled['age'] * weights['age'] + df_scaled['BFP'] * weights['BFP'] + df_scaled['train_count'] * weights['train_count']

# 그래프 생성
plt.figure(figsize=(12, 6))  # 전체 그림 크기 설정

# K-평균 클러스터링 수행 및 'Grade vs. Total' 그래프 생성
kmeans_total = KMeans(n_clusters=3, random_state=0)
df_scaled['cluster_total'] = kmeans_total.fit_predict(df_scaled[['total']])
plt.subplot(1, 2, 1)  # 1행 2열의 첫 번째 위치에 subplot 생성
plt.scatter(df_scaled['total'], df['grade'], c=df_scaled['cluster_total'], cmap='viridis')
plt.xlabel('Total')
plt.ylabel('Grade')
plt.title('Grade vs. Total')
plt.colorbar(label='Cluster ID')

# K-평균 클러스터링 수행 및 'Grade vs. Train Count' 그래프 생성 (정규화되지 않은 train_count 사용)
kmeans_train_count = KMeans(n_clusters=3, random_state=0)
df['cluster_train_count'] = kmeans_train_count.fit_predict(df[['train_count']])
plt.subplot(1, 2, 2)  # 1행 2열의 두 번째 위치에 subplot 생성
plt.scatter(df['train_count'], df['grade'], c=df['cluster_train_count'], cmap='viridis')
plt.xlabel('Train Count')
plt.ylabel('Grade')
plt.title('Grade vs. Train Count')
plt.colorbar(label='Cluster ID')

plt.tight_layout()  # subplot 간격 자동 조정
plt.show()