import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

"""
# 1. 데이터 읽기
df = pd.read_csv('C:\\dev\\koreatech\\graduationProject\\kmeans\\efc\\재활훈련정보.csv')
"""

# 1-1. 더미데이터 test
df = pd.read_csv('C:\\dev\\koreatech\\graduationProject\\kmeans\\efc\\수동더미데이터.csv')

# 2. 필요한 열 선택 및 결측치 제거
df = df[['grade', 'height', 'age', 'BFP', 'train_count']].dropna()

# 3. 데이터 표준화
X = df[['height', 'age', 'BFP', 'train_count']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. K-Means 클러스터링 적용
kmeans = KMeans(n_clusters=7, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 5. 클러스터링 결과를 기반으로 2x2 그리드에 산점도 그리기
fig, axs = plt.subplots(2, 2, figsize=(15, 10)) # 2x2 그리드 생성

# 각 변수에 대한 산점도 그리기
variables = ['height', 'age', 'BFP', 'train_count']
for i, var in enumerate(variables):
    row = i // 2  # 행 인덱스 결정
    col = i % 2   # 열 인덱스 결정
    sns.scatterplot(data=df, x=var, y='grade', hue='cluster', palette='tab10', ax=axs[row, col])
    axs[row, col].set_title(f'Grade vs {var}')  # 각 그래프의 제목 설정

# 범례 위치가 자동으로 설정되므로, plt.legend(loc='upper right') 호출은 제거합니다.
plt.tight_layout()  # 그래프 간격 조정
plt.show()