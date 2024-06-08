import joblib
import numpy as np
import pandas as pd
import math

# 모델을 불러옵니다.
model1 = joblib.load('trained_model_with_balance.pkl')
model2 = joblib.load('trained_model_with_balance2.pkl')

# 환자의 정보를 입력받습니다.
patient_info = np.array([[170, 35, 1, 20, 5]])

# 모델 학습에 사용된 특성 이름 확인
feature_names_model1 = ['Height_cm', 'Age_years', 'Gender', 'BodyFat_Percent', 'Balance_Activity']
feature_names_model2 = ['Height_cm', 'Age_years', 'Gender', 'BodyFat_Percent', 'Train_count']

# 환자 정보를 DataFrame으로 변환, 초기에는 model2에 맞게 설정
patient_info_df = pd.DataFrame(patient_info, columns=feature_names_model2)

# 예측하고자 하는 균형활성도
prediction = float(input("예측하고자 하는 균형활성도를 입력하세요: "))

# 환자 정보 중 훈련 횟수의 인덱스, 이 예시에서는 마지막 요소를 의미합니다.
frequency_index = -1

# 기존 훈련 횟수 저장
original_frequency = patient_info[0][frequency_index]
print(f"현재 훈련 횟수 : {original_frequency}회")

case = 0

while True:
    if patient_info[0][frequency_index] > 150:
        # 훈련 횟수가 150을 넘어가면 model1을 사용합니다.
        # 균형활성도를 기준으로 훈련 횟수를 예측하기 위해 데이터프레임을 재구성합니다.
        patient_info_df = pd.DataFrame(patient_info, columns=feature_names_model1)
        patient_info_df['Balance_Activity'] = prediction  # 예측하고자 하는 균형활성도를 추가합니다.
        current_prediction = model1.predict(patient_info_df)[0]
        case = 2
        break  # 모델 1을 사용하면 바로 반복문을 종료합니다.
    else:
        # 모델 2를 사용하여 훈련 횟수를 예측합니다.
        current_prediction = model2.predict(patient_info_df)[0]
        current_prediction_ceil = math.ceil(current_prediction)
        patient_info_df = pd.DataFrame(patient_info, columns=feature_names_model2)  # 모델2용 데이터프레임

    # 예측값이 입력받은 prediction과 일치하는지 확인합니다.
    if current_prediction_ceil == prediction:
        #print(f"예측 균형활성도: {current_prediction_ceil}, 예상 훈련 횟수: {patient_info[0][frequency_index]}회")
        case = 1
        break
    else:
        # 훈련 횟수를 1 증가시키고 데이터프레임을 갱신합니다.
        patient_info[0][frequency_index] += 1
        patient_info_df = pd.DataFrame(patient_info, columns=feature_names_model2)
#remain=0
# 최종적으로 예측된 훈련 횟수와 균형활성도를 출력합니다.
if case == 1:
    print(f"모델 2를 사용하여 예측된 균형활성도: {current_prediction_ceil}, 예상 훈련 횟수: {patient_info[0][frequency_index]}회")
    remain = patient_info[0][frequency_index] - original_frequency
    print(f"남은 훈련 횟수: {remain}회")
elif case == 2:
    print(f"모델 1을 사용하여 예측된 훈련 횟수: {math.ceil(current_prediction)}회, 균형활성도 목표: {math.ceil(prediction)}")
    remain = math.ceil(current_prediction) - original_frequency
    print(f"남은 훈련 횟수: {remain}회")
else:
    print("적절한 예측을 찾지 못했습니다.")
if (remain - original_frequency > 100):
    print("예상 훈련 횟수가 100회를 넘습니다. 훈련 강도를 한단계 높여보세요.")

