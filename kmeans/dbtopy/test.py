import sqlite3
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 데이터베이스 연결 및 데이터 읽기
conn = sqlite3.connect('storedb.db')  # 데이터베이스 파일 경로
query = "SELECT * FROM user"  # 'user' 테이블에서 모든 데이터를 선택
df = pd.read_sql_query(query, conn)
conn.close()

# 데이터 전처리
df.dropna(inplace=True)  # 결측치 있는 행 제거
features = df[['Height', 'BFP']]  # 'Height'와 'BFP' 열을 사용할 특성으로 선택
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# K-means 알고리즘 적용
kmeans = KMeans(n_clusters=3)  # 예: 3개의 클러스터로 구분
kmeans.fit(features_scaled)
df['cluster'] = kmeans.labels_

# 결과 확인
print(df.head())

# 추가 분석 및 시각화 등