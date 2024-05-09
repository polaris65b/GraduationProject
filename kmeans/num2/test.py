# 한글 설정
import matplotlib.pyplot as plt

print(plt.rcParams['font.family'])
plt.rcParams['font.family'] = 'Malgun Gothic'

#Step 1. 분석할 데이터가 저장된 파일을 불러와서 변수에 할당
#from google.colab import files
#myfile = files.upload()

import io
import pandas as pd
#pd.read_csv로 csv파일 불러오기
patients = pd.read_csv('C:\dev\koreatech\graduationProject\kmeans\임시_환자정보.csv', sep=",",header=0, encoding='cp949')
patients

#Step 2. 데이터의 분포를 그림으로 그리고 임의의 중심점 지정
import matplotlib.pyplot as plt
x1, y1 = 200, 40
x2, y2 = 175, 50
x3, y3 = 150, 65
x4, y4 = 125, 80

data = patients[['신장(cm)', '측정개수변화량']]
plt.figure(figsize=(7, 5))
plt.title("Before", fontsize=15)
plt.plot(data["신장(cm)"], data["측정개수변화량"], "o", label="Data")
plt.plot([x1, x2, x3, x4], [y1, y2, y3, y4], "rD", \
         marker='*', markersize=12, label="init_Centroid")
plt.xlabel("신장(cm)", fontsize=12)
plt.ylabel("측정개수변화량", fontsize=12)
plt.legend()
plt.grid()
plt.show()


#Step 3. 군집 분석 수행
from sklearn.cluster import KMeans
import numpy as np
data = patients[["신장(cm)", "측정개수변화량"]]

# 초기의 점을 지정한 경우
kmeans = KMeans(n_clusters=4, init=np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)]))

# 초기의 점을 지정하지 않은 경우
#kmeans = KMeans(n_clusters=3)
kmeans.fit(data)
data['cluster'] = kmeans.labels_
final_centroid = kmeans.cluster_centers_



data


#Step 4. 군집화를 진행하여 최종 결과를 확인
plt.figure(figsize=(7, 5))
plt.title("After", fontsize=15)
plt.scatter(data["신장(cm)"], data["측정개수변화량"], c=data['cluster'])
plt.plot(final_centroid[:, 0], final_centroid[:, 1], "rD", \
         marker='*', markersize=12, label="final_Centroid")
plt.xlabel("신장(cm)", fontsize=12)
plt.ylabel("측정개수변화량", fontsize=12)
plt.legend()
plt.grid()
plt.show()