import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# 데이터 불러오기
data = pd.read_csv('재활훈련정보1.csv', encoding='cp949')

# 데이터 확인
print(data.head())

# 1. 시각화를 통한 비선형성 확인

## Pairplot을 이용하여 변수 간의 관계 시각화
sns.pairplot(data)
plt.show()

## 나이와 균형활성도 간의 scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='나이(세)', y='균형활성도', data=data)
plt.title('나이와 균형활성도 간의 관계')
plt.xlabel('나이(세)')
plt.ylabel('균형활성도')
plt.show()

# 2. Spearman 순위 상관 계수를 이용한 비선형 상관 관계 분석

## 나이와 균형활성도 간의 Spearman 순위 상관 계수
corr, p_value = spearmanr(data['나이(세)'], data['균형활성도'])
print(f'나이와 균형활성도 간의 Spearman 순위 상관 계수: {corr:.3f}, p-value: {p_value:.3f}')

## 나이와 훈련 횟수 간의 Spearman 순위 상관 계수
corr, p_value = spearmanr(data['나이(세)'], data['훈련 횟수'])
print(f'나이와 훈련 횟수 간의 Spearman 순위 상관 계수: {corr:.3f}, p-value: {p_value:.3f}')