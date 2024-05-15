import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 데이터 로딩
df = pd.read_csv('C:\dev\koreatech\graduationProject\kmeans\efc\재활훈련정보.csv')

# 각 user_id에 대한 마지막 행을 선택합니다(가장 큰 train_id를 가진 행).
df = df.sort_values('train_id').groupby('user_id').last().reset_index()

# 사용할 열을 선택합니다.
df = df[['height', 'age', 'BFP', 'grade']]

# 데이터 정규화
scaler = StandardScaler()
df[['height', 'age', 'BFP']] = scaler.fit_transform(df[['height', 'age', 'BFP']])

# K-평균 클러스터링 수행
kmeans = KMeans(n_clusters=3, random_state=0)
df['cluster'] = kmeans.fit_predict(df[['height', 'age', 'BFP']])

# 시각화를 위해 PCA 수행
pca = PCA(n_components=2)
df['x'] = pca.fit_transform(df[['height', 'age', 'BFP']])[:, 0]
df['y'] = pca.fit_transform(df[['height', 'age', 'BFP']])[:, 1]

# 결과를 그래프로 표현
plt.scatter(df['x'], df['y'], c=df['cluster'])
plt.show()