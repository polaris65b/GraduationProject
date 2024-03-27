import sqlite3

def read_from_db(db_file):
    try:
        # SQLite 데이터베이스에 연결
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # 예시 쿼리: 모든 테이블의 데이터를 가져옴
        cursor.execute("SELECT ID, Name, Birth, Sex, Height, BFP FROM user;")
        tables = cursor.fetchall()

        for table in tables:
            table_name = table[0]
            print(f"Table: {table_name}")
            
            # 각 테이블의 데이터를 읽어옴
            cursor.execute(f"SELECT * FROM {table_name};")
            rows = cursor.fetchall()

            # 결과 출력
            for row in rows:
                print(row)

    except sqlite3.Error as e:
        print("SQLite 오류:", e)
    finally:
        if conn:
            conn.close()

# 데이터베이스 파일 경로
db_file_path = 'storedb.db'

# 함수 호출
read_from_db(db_file_path)