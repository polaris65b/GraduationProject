import os
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import joblib
from matplotlib.ticker import MaxNLocator

# 데이터 로딩
df = pd.read_csv("2.csv", encoding='cp949')

# 각 user_id에 대한 마지막 행을 선택
df = df.sort_values('train_id').groupby('user_id', as_index=False).last()

# 사용할 열을 선택합니다
df = df[['user_id', 'height', 'age', 'BFP', 'grade', 'train_count', 'total']]

# 특성 리스트
features = ['height', 'age', 'BFP', 'train_count']

# 환자의 user_id 입력
user_id = int(input("환자의 user_id를 입력하세요: "))

# 해당 환자의 데이터 찾기
patient_data = df[df['user_id'] == user_id]

if not patient_data.empty:
    # 환자의 모든 정보를 numpy 배열로 저장 (마지막 'total' 컬럼 제외)
    patient_info=patient_data.iloc[0]
    print(patient_info)
    patient_info_without_total = patient_data.iloc[0].values[:-2]  # 'total' 제외
    print(patient_info_without_total)
    patient_info2 = np.array([patient_info_without_total])  # 2D 배열로 변환
    print(patient_info2)
    patient_info_without_grade_and_total = np.concatenate((patient_info[:-3], patient_info[-2:-1]))
    print(patient_info_without_grade_and_total)
    patient_info3 = np.array([patient_info_without_grade_and_total])  # 2D 배열로 변환
    print(patient_info3)

    print("환자 정보:", patient_info)
    print("환자 정보(‘total’ 제외):", patient_info2)
    print("환자 정보(‘total’ 제외):", patient_info3)

    # 입력받은 user_id의 total 찾기
    user_total = patient_data['total'].values[0]

    # total과의 차이 계산
    df['total_diff'] = abs(df['total'] - user_total)

    # total 차이가 가장 작은 상위 5명 찾기
    closest_users = df.sort_values('total_diff').head(6)  # 입력받은 사용자 포함 상위 6명
else:
    print("해당 user_id를 가진 환자를 찾을 수 없습니다.")

# 모델 불러오기
model1 = joblib.load('trained_model_with_balance.pkl')
model2 = joblib.load('trained_model_with_balance2.pkl')

feature_names_model1 = ['Height_cm', 'Age_years', 'Gender', 'BodyFat_Percent', 'Balance_Activity']
feature_names_model2 = ['Height_cm', 'Age_years', 'Gender', 'BodyFat_Percent', 'Train_count']
patient_info_df = pd.DataFrame(patient_info2, columns=feature_names_model1)
Max_Train = np.round(model1.predict(patient_info_df)[0])+20

patient_info_df = pd.DataFrame(patient_info3, columns=feature_names_model2)

train_counts = []
balance_activities = []

# patient_info3의 마지막 항목에서 train_counts의 첫 번째 값 추가
last_train_count = patient_info3[-1][-1]
train_counts.append(last_train_count)

# patient_info2의 마지막 항목에서 balance_activities의 첫 번째 값 추가
last_balance_activity = patient_info2[-1][-1]
balance_activities.append(last_balance_activity)

frequency_index = -1
original_frequency = patient_info3[0][frequency_index]

# 첫 번째 실행을 건너뛰고 두 번째 실행부터 시작
patient_info3[0][frequency_index] += 1

while patient_info3[0][frequency_index] < Max_Train:
    current_prediction = np.round(model2.predict(patient_info_df)[0])

    if current_prediction <= 1:
        break

    train_counts.append(patient_info3[0][frequency_index])
    balance_activities.append(current_prediction)

    patient_info3[0][frequency_index] += 1
    patient_info_df = pd.DataFrame(patient_info3, columns=feature_names_model2)

# 입력받은 사용자 제외
closest_users = closest_users[closest_users['user_id'] != user_id]

plt.figure(figsize=(18, 12))  # 전체 그림 크기 설정

for i, feature in enumerate(features, 1):
    # K-평균 클러스터링 수행
    kmeans = KMeans(n_clusters=3, random_state=0)
    df['cluster'] = kmeans.fit_predict(df[[feature]])

    plt.subplot(2, 2, i)
    plt.scatter(df[feature], df['grade'], c=df['cluster'], cmap='viridis')
    plt.xlabel(feature)
    plt.ylabel('Grade')
    plt.title(f'Grade vs. {feature}')
    plt.colorbar(label='Cluster ID')

    # 입력된 user_id의 군집 ID 찾기
    user_cluster_id = df[df['user_id'] == user_id]['cluster'].iloc[0]
    # 동일한 군집에 속한 데이터의 grade 값들을 추출
    cluster_grades = df[df['cluster'] == user_cluster_id]['grade']
    # 입력된 user_id의 grade 값
    user_grade = patient_data['grade'].iloc[0]
    # 상위 몇 %에 해당하는지 계산
    percentile = (np.sum(cluster_grades <= user_grade) / len(cluster_grades)) * 100

    if feature == 'train_count':


        plt.plot(train_counts, balance_activities, marker='o')

        # 동일한 grade 값을 가진 데이터 필터링
        same_grade_df = df[df['grade'] == user_grade]
        # train_count 값의 분포 확인
        train_count_values = same_grade_df['train_count'].values
        # 입력된 user_id의 train_count 값
        user_train_count = patient_data['train_count'].iloc[0]
        # 상위 몇 %에 해당하는지 계산
        percentile = (np.sum(train_count_values <= user_train_count) / len(train_count_values)) * 100
        plt.scatter(patient_data[feature], patient_data['grade'], c='red', s=200, marker='*', edgecolors='black', 
                    label=f'User ID {user_id} (Top {percentile:.2f}%)')
        
            # total 값 차이가 가장 작은 5명의 환자도 표시
        for _, row in closest_users.iterrows():
            plt.scatter(row[feature], row['grade'], c='blue', s=100, marker='*', edgecolors='black', 
                    label=f'Closest User ID {row["user_id"]}')
    else:
        plt.scatter(patient_data[feature], patient_data['grade'], c='red', s=200, marker='*', edgecolors='black', 
                label=f'User ID {user_id} (Top {percentile:.2f}%) in Cluster')



    plt.legend()

plt.tight_layout()  # subplot 간격 자동 조정
plt.show()
