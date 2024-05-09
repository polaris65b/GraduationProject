import sqlite3

def fetch_data(table_name):
    # Connect to the SQLite DB file
    conn = sqlite3.connect('C:\dev\koreatech\graduationProject\kmeans\dbtopy\storedb.db')
    cursor = conn.cursor()
    
    # Build the SQL query string
    query = f"SELECT * FROM {table_name}"
    
    # Execute the query
    cursor.execute(query)
    rows = cursor.fetchall()
    
    # Print the results
    for row in rows:
        print(row)
    
    # Close the connection
    conn.close()

def main():
    print("어떤 테이블의 데이터를 불러올까요?")
    print("1: user 테이블")
    print("2: history 테이블")
    
    choice = input("입력 (1 또는 2): ")
    
    if choice == '1':
        fetch_data('user')
    elif choice == '2':
        fetch_data('history')
    else:
        print("잘못된 선택입니다. 1 또는 2를 입력해주세요.")

if __name__ == '__main__':
    main()