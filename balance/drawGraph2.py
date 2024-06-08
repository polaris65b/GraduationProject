import joblib
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# 모델 불러오기
model1 = joblib.load('trained_model_with_balance.pkl')
model2 = joblib.load('trained_model_with_balance2.pkl')

#예시 환자 정보
patient_info = np.array([[163, 70, 1, 25, 5]])  


feature_names_model1 = ['Height_cm', 'Age_years', 'Gender', 'BodyFat_Percent', 'Balance_Activity']
feature_names_model2 = ['Height_cm', 'Age_years', 'Gender', 'BodyFat_Percent', 'Train_count']
patient_info2 = np.array([[170, 35, 1, 20, 1]])
patient_info_df = pd.DataFrame(patient_info2, columns=feature_names_model1)
Max_Train = np.round(model1.predict(patient_info_df)[0])+20

patient_info_df = pd.DataFrame(patient_info, columns=feature_names_model2)


train_counts = []
balance_activities = []
frequency_index = -1
original_frequency = patient_info[0][frequency_index]

while patient_info[0][frequency_index] < Max_Train:
    current_prediction = np.round(model2.predict(patient_info_df)[0])

    if current_prediction <= 1:
        break

    train_counts.append(patient_info[0][frequency_index])
    balance_activities.append(current_prediction)

    patient_info[0][frequency_index] += 1
    patient_info_df = pd.DataFrame(patient_info, columns=feature_names_model2)

from matplotlib.ticker import MaxNLocator

plt.plot(train_counts, balance_activities, marker='o')
plt.title('Trend of Predicted Balance Activity')
plt.xlabel('Training Count')
plt.ylabel('Balance Activity')
plt.grid(True)

# X축에 대해 정수만 표시되도록 설정
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.show()
