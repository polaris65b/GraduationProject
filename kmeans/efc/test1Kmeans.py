import numpy as np
import matplotlib.pyplot as plt

def kmeans(X, k, max_iters=100):
    # 데이터 초기화
    centroids = X[np.random.choice(range(len(X)), k, replace=False)]
    
    for _ in range(max_iters):
        # 각 데이터 포인트에 대해 가장 가까운 중심 찾기
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)
        
        # 새로운 중심 계산
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        
        # 중심 업데이트
        centroids = new_centroids
        
    return centroids, labels

# 테스트 데이터 생성
np.random.seed(42)
data = np.concatenate([np.random.normal(loc, 1, size=(100, 2)) for loc in [(0, 0), (5, 5), (10, 10)]])
np.random.shuffle(data)

# K-means 알고리즘 적용
k = 3
centroids, labels = kmeans(data, k)

# 결과 시각화
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.title('K-means Clustering')
plt.legend()
plt.show()