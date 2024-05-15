import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# 1. 데이터 읽기
df = pd.read_csv('C:\\dev\\koreatech\\graduationProject\\kmeans\\efc\\재활훈련정보.csv')

# 2. 필요한 열 선택 및 결측치 제거
df = df[['grade', 'height', 'age', 'BFP', 'train_count']].dropna()

# 3. 데이터 표준화
X = df[['height', 'age', 'BFP', 'train_count']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. K-Means 클러스터링 적용
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 5. 클러스터링 결과 시각화
# PairGrid를 사용하여 시각화 커스터마이징
g = sns.PairGrid(df, vars=['height', 'age', 'BFP', 'train_count'], hue='cluster', palette='viridis')
# 대각선에는 그래프를 그리지 않음
g = g.map_diag(sns.scatterplot, edgecolor="w")  # 대각선 그래프를 그리고 싶지 않다면 이 줄을 제거
# 대각선 외의 부분에는 산점도 그래프를 그림
g = g.map_offdiag(sns.scatterplot)
# 범례 추가
g = g.add_legend()

plt.show()