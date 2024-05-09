import sqlite3
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.impute import SimpleImputer
import numpy as np

# Database path 설정
db_path = 'C:\dev\koreatech\graduationProject\kmeans\dbtopy\storedb.db'

# 데이터베이스에서 데이터를 읽어와서 DataFrame으로 결합하는 함수
def fetch_joined_data(db_path):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT u.ID, u.Birth, u.Sex, u.Height, u.BFP, h.Grade
    FROM user u
    JOIN history h ON u.ID = h.user_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

#데이터 전 처리
def preprocess_data(df):
    # 성별을 숫자로 인코딩
    df['Sex'] = df['Sex'].map({'남성': 0, '여성': 1})
    
    # 'Birth' 열을 datetime 객체로 변환
    df['Birth'] = pd.to_datetime(df['Birth'])
    
    # 기준일 설정
    reference_date = datetime(2024, 5, 10)
    
    # 'Birth' 열에서 각 사용자가 살아온 일수 계산
    df['Days_Lived'] = (reference_date - df['Birth']).dt.days
    
    # NaN 값 처리
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    df[['Height', 'BFP', 'Grade']] = imputer.fit_transform(df[['Height', 'BFP', 'Grade']])
    
    # Drop rows with missing values
    df.dropna(inplace=True)
    
    if len(df) == 0:
        raise ValueError("No samples remaining after dropping missing values.")
    
    # 스케일링
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[['Days_Lived', 'Height', 'BFP', 'Grade', 'Sex']])
    
    df_scaled = pd.DataFrame(scaled_features, columns=['Days_Lived', 'Height', 'BFP', 'Grade', 'Sex'])
    
    return df_scaled

# K-means 클러스터링 실행
def apply_kmeans(df, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df)
    return df, kmeans.cluster_centers_

# 메인 함수
def main():
    df = fetch_joined_data(db_path)
    df_preprocessed = preprocess_data(df)
    df_clustered, centers = apply_kmeans(df_preprocessed, n_clusters=3)
    
    print(df_clustered.head())
    print("클러스터 센터:", centers)

if __name__ == '__main__':
    main()