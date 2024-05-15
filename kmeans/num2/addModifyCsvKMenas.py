import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# 데이터 읽기 (더미데이터 사용)
df = pd.read_csv('C:\\dev\\koreatech\\graduationProject\\kmeans\\efc\\수동더미데이터.csv')

# 필요한 열 선택 및 결측치 제거
df = df[['user_id', 'grade', 'height', 'age', 'BFP', 'train_count', 'train_id']].dropna()

# 데이터 표준화
X = df[['height', 'age', 'BFP', 'train_count']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means 클러스터링 적용
kmeans = KMeans(n_clusters=3, random_state=40)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 사용자로부터 user_id 입력받기
# 예시: user_id = input("Enter user_id: ") 
user_id = '3'  # 예시 사용자 ID, 실제 사용 시 입력받은 값으로 대체

# 입력받은 user_id에 대해 train_id가 숫자적으로 가장 높은 데이터 찾기
highlight = df[df['user_id'] == user_id].nlargest(1, 'train_id')

# 클러스터링 결과를 기반으로 2x2 그리드에 산점도 그리기
fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # 2x2 그리드 생성

# 각 변수에 대한 산점도 그리기1
variables = ['height', 'age', 'BFP', 'train_count']
for i, var in enumerate(variables):
    row = i // 2  # 행 인덱스 결정
    col = i % 2   # 열 인덱스 결정
    sns.scatterplot(data=df, x=var, y='grade', hue='cluster', palette='tab10', ax=axs[row, col])
    axs[row, col].set_title(f'Grade vs {var}')  # 각 그래프의 제목 설정
    
    # 가장 최근 훈련 세션 (train_id가 가장 높은 데이터) 별표시로 강조
    if not highlight.empty:
        axs[row, col].scatter(highlight[var], highlight['grade'], s=100, color='red', marker='*')

plt.tight_layout()  # 그래프 간격 조정
plt.show()