def read_data_and_calculate(filename):
    level = int(input("레벨을 입력하세요 (1, 2, 3): "))#1단계가 판을 하나도 넣지 않은 제일 쉬운 단계(7~10)/2단계가 판을 하나 넣은 중간 단계(4~6)/3단계가 판을 두 개 넣은 어려운 단계(1~3)
    limits = {1: 10, 2: 15, 3: 20}#단계별로 한계치를 10,15,20으로 설정
    limit = limits.get(level, 10)
    ground_count = 0
    max_time_under_5 = 0
    current_time_under_5 = 0

    with open(filename, 'r') as file:
        for line in file:
            x, y = map(float, line.split(','))
            if abs(x) > limit or abs(y) > limit:
                ground_count += 1
            if abs(x) <= 5 and abs(y) <= 5:
                current_time_under_5 += 1
            else:
                max_time_under_5 = max(max_time_under_5, current_time_under_5)
                current_time_under_5 = 0


    max_time_under_5 = max(max_time_under_5, current_time_under_5)

    print(f"limit를 초과한 횟수: {ground_count}")#땅에 닿은 횟수
    print(f"가로 데이터와 세로 데이터의 절대값이 모두 5를 넘지 않은 가장 긴 시간: {max_time_under_5}초")#안정적으로 유지한 시간

read_data_and_calculate('수민_table_data_2024_03_13_15_22_49.txt')