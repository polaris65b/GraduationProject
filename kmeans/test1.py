import sqlite3
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 데이터베이스 연결 및 데이터 로딩
conn = sqlite3.connect('storedb.db')
query = """
SELECT u.Birth, u.Sex, u.Height, u.BFP, h.Grade
FROM user u
JOIN history h ON u.UserID = h.UserID
"""
df = pd.read_sql_query(query, conn)
conn.close()

# 데이터 전처리 (예시: Sex를 숫자로 인코딩, 표준화 등)
# 이 부분은 데이터에 따라 달라지므로 적절한 전처리가 필요합니다.
# 여기서는 Height와 BFP만 사용한다고 가정하겠습니다.
X = df[['Height', 'BFP']].values
X = StandardScaler().fit_transform(X)

# K-means 클러스터링 수행
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
df['Cluster'] = kmeans.labels_

# 시각화
fig, ax = plt.subplots()
for cluster in df['Cluster'].unique():
    cluster_data = df[df['Cluster'] == cluster]
    ax.scatter(cluster_data['Height'], cluster_data['Grade'], label=f'Cluster {cluster}')
ax.set_xlabel('Height')
ax.set_ylabel('Grade')
ax.legend()
plt.show()
