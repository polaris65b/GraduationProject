import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 데이터 읽기
df = pd.read_csv('C:\dev\koreatech\graduationProject\kmeans\efc\재활훈련정보.csv')

# 2. 전처리: 필요한 컬럼만 선택하고, 누락된 값이 있는 행을 제거
df = df[['grade', 'height', 'age', 'BFP', 'train_count']].dropna()

# 3. x축 데이터에 해당하는 컬럼들만 선택하여 표준화
X = df[['height', 'age', 'BFP', 'train_count']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. K-Means 클러스터링 적용
kmeans = KMeans(n_clusters=3, random_state=42)  # n_clusters는 적절한 클러스터 수로 조정해야 함
df['cluster'] = kmeans.fit_predict(X_scaled)

# 5. 시각화
# PCA를 사용하여 데이터를 2차원으로 축소
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df['pca_x'] = X_pca[:, 0]
df['pca_y'] = X_pca[:, 1]

# seaborn을 사용하여 클러스터링 결과 시각화
plt.figure(figsize=(10, 7))
sns.scatterplot(x='pca_x', y='pca_y', hue='cluster', data=df, palette='viridis', alpha=0.7)
plt.title('K-Means 클러스터링 결과')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()