# -*- coding: utf-8 -*-
import os
import sqlite3
from datetime import datetime

import paramiko
# Form implementation generated from reading ui file 'Train_start_result.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QDate, QPointF
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QMessageBox
from PyQt5.QtGui import QPainter, QBrush, QColor
from PyQt5.QtCore import Qt, QPoint, QPointF
from PyQt5.QtGui import QColor, QPen
import Home
import Calinder
import Train
import user_inform_modify
import Login

result_grade = 0

class TargetWidget(QWidget):
    def __init__(self, x_list, y_list):
        super().__init__()
        self.initUI()
        self.x_list = x_list
        self.y_list = y_list

    def initUI(self):
        self.setGeometry(100, 100, 450, 450)
        self.setWindowTitle('Target Widget')

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        self.drawTarget(qp)
        qp.end()

    def drawTarget(self, qp):
        # Draw target
        size = self.size()
        center = QPoint(size.width() // 2, size.height() // 2)
        maxRadius = min(size.width(), size.height()) // 2
        numRings = 3
        ringWidth = maxRadius / (2 * numRings)

        # 각 원의 반지름 리스트 설정
        radius_list = [maxRadius, maxRadius - 2 * ringWidth, maxRadius - 4 * ringWidth]

        colors = [
            QColor(255, 204, 204),  # 연한 분홍색
            QColor(255, 255, 204),  # 연한 노란색
            QColor(204, 229, 255)  # 연한 파란색
        ]  # 초록색

        for i in range(numRings):
            radius = radius_list[i]  # 개별 원의 반지름 지정
            color = colors[i]
            qp.setPen(Qt.NoPen)
            qp.setBrush(QBrush(color))
            qp.drawEllipse(center, radius, radius)

        # x축 그리기
        pen = QPen(Qt.black)
        qp.setPen(pen)
        qp.drawLine(0, center.y(), size.width(), center.y())

        # y축 그리기
        qp.drawLine(center.x(), 0, center.x(), size.height())

        self.drawPoint(qp)

    def drawPoint(self, qp):
        # 점의 크기
        point_size = 5

        for x, y in zip(self.x_list, self.y_list):
            # 점 그리기
            qp.drawEllipse(QPointF(x * 7 + self.width() / 2, self.height() / 2 - (y * 7)), point_size, point_size)


class Train_start_result(object):
    def __init__(self, main_window):
        self.MainWindow = main_window

    def setupUi(self, MainWindow, userID, trainList):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1287, 720)
        MainWindow.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.save_button = QtWidgets.QPushButton(self.centralwidget)
        self.save_button.setGeometry(QtCore.QRect(280, 560, 231, 111))
        self.save_button.setStyleSheet("background-color:rgba(0, 0, 0, 0);")
        self.save_button.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("images/save.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.save_button.setIcon(icon)
        self.save_button.setIconSize(QtCore.QSize(190, 150))
        self.save_button.setObjectName("save_button")
        # self.result_table = QtWidgets.QTableView(self.centralwidget)
        self.result_table = QtWidgets.QScrollArea(self.centralwidget)
        self.result_table.setGeometry(QtCore.QRect(750, 150, 471, 491))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.result_table.setFont(font)
        self.result_table.setStyleSheet("background-color:rgb(255, 255, 255);border-radius:235;\n"
"border:1px solid rgb(204, 204, 204);")
        self.result_table.setObjectName("result_table")
        self.start_button_2 = QtWidgets.QPushButton(self.centralwidget)
        self.start_button_2.setGeometry(QtCore.QRect(50, 510, 161, 161))
        self.start_button_2.setStyleSheet("background-color:rgba(0, 0, 0, 0);")
        self.start_button_2.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("images/icon_btn_start_hover.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.start_button_2.setIcon(icon1)
        self.start_button_2.setIconSize(QtCore.QSize(150, 150))
        self.start_button_2.setObjectName("start_button_2")
        self.train_background_label = QtWidgets.QLabel(self.centralwidget)
        self.train_background_label.setGeometry(QtCore.QRect(40, 580, 201, 91))
        self.train_background_label.setStyleSheet("background-color: rgba(255, 255, 255, 100);")
        self.train_background_label.setText("")
        self.train_background_label.setObjectName("train_background_label")
        self.date_label = QtWidgets.QLabel(self.centralwidget)
        self.date_label.setGeometry(QtCore.QRect(560, 40, 331, 61))
        font = QtGui.QFont()
        font.setPointSize(25)
        font.setBold(True)
        font.setWeight(75)
        self.date_label.setFont(font)
        self.date_label.setStyleSheet("border: 1px solid rgb(200,200,200); \n"
"border-radius:5px;")
        self.date_label.setAlignment(QtCore.Qt.AlignCenter)
        self.date_label.setObjectName("date_label")
        self.home_background_label = QtWidgets.QLabel(self.centralwidget)
        self.home_background_label.setGeometry(QtCore.QRect(50, 160, 191, 41))
        self.home_background_label.setStyleSheet("background-color: rgba(255, 255, 255, 100);")
        self.home_background_label.setText("")
        self.home_background_label.setObjectName("home_background_label")
        self.result_background_label = QtWidgets.QLabel(self.centralwidget)
        self.result_background_label.setGeometry(QtCore.QRect(50, 230, 191, 41))
        self.result_background_label.setStyleSheet("background-color: rgba(255, 255, 255, 100);")
        self.result_background_label.setText("")
        self.result_background_label.setObjectName("result_background_label")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(270, 0, 1021, 721))
        self.label_4.setStyleSheet("background-color:rgb(238, 238, 238);")
        self.label_4.setText("")
        self.label_4.setObjectName("label_4")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(-30, 0, 1201, 720))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("images/background (2).png"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 40, 201, 61))
        self.label_2.setStyleSheet("border-radius:10px;\n"
"background-color: rgba(157, 157, 157, 45)")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        # self.user_label = QtWidgets.QLabel(self.centralwidget)
        self.user_label = QtWidgets.QPushButton(self.centralwidget)
        self.user_label.setGeometry(QtCore.QRect(100, 55, 81, 31))
        font = QtGui.QFont()
        font.setFamily("Noto Sans KR Medium")
        font.setPointSize(18)
        font.setBold(False)
        font.setWeight(50)
        self.user_label.setFont(font)
        self.user_label.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        #self.user_label.setAlignment(QtCore.Qt.AlignCenter)
        self.user_label.setObjectName("user_label")
        self.user_image_label = QtWidgets.QLabel(self.centralwidget)
        self.user_image_label.setGeometry(QtCore.QRect(40, 50, 41, 41))
        self.user_image_label.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.user_image_label.setText("")
        self.user_image_label.setPixmap(QtGui.QPixmap("images/user (1).png"))
        self.user_image_label.setScaledContents(True)
        self.user_image_label.setObjectName("user_image_label")
        self.home_button = QtWidgets.QPushButton(self.centralwidget)
        self.home_button.setGeometry(QtCore.QRect(40, 220, 161, 31))
        self.home_button.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.home_button.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("images/home.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.home_button.setIcon(icon2)
        self.home_button.setIconSize(QtCore.QSize(190, 50))
        self.home_button.setObjectName("home_button")
        self.result_button = QtWidgets.QPushButton(self.centralwidget)
        self.result_button.setGeometry(QtCore.QRect(40, 280, 161, 31))
        self.result_button.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.result_button.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("images/trainresult.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.result_button.setIcon(icon3)
        self.result_button.setIconSize(QtCore.QSize(190, 50))
        self.result_button.setObjectName("result_button")
        self.logout_button = QtWidgets.QPushButton(self.centralwidget)
        self.logout_button.setGeometry(QtCore.QRect(30, 335, 191, 51))
        self.logout_button.setStyleSheet("background-color:rgba(255, 255, 255, 0);")
        self.logout_button.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("images/logout.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.logout_button.setIcon(icon)
        self.logout_button.setIconSize(QtCore.QSize(190, 50))
        self.logout_button.setObjectName("logout_button")
        self.fix_label1 = QtWidgets.QGroupBox(self.centralwidget)
        self.fix_label1.setGeometry(QtCore.QRect(300, 150, 181, 181))
        self.fix_label1.setStyleSheet("border-radius:20px;border:1px solid rgb(204, 204, 204);")
        self.fix_label1.setTitle("")
        self.fix_label1.setObjectName("fix_label1")
        self.fix_label1_1 = QtWidgets.QLabel(self.fix_label1)
        self.fix_label1_1.setGeometry(QtCore.QRect(30, 20, 121, 31))
        font = QtGui.QFont()
        font.setFamily("Noto Sans KR Medium")
        font.setPointSize(20)
        self.fix_label1_1.setFont(font)
        self.fix_label1_1.setStyleSheet("border:none;")
        self.fix_label1_1.setAlignment(QtCore.Qt.AlignCenter)
        self.fix_label1_1.setObjectName("fix_label1_1")
        self.label_5 = QtWidgets.QLabel(self.fix_label1)
        self.label_5.setGeometry(QtCore.QRect(40, 70, 81, 71))
        font = QtGui.QFont()
        font.setFamily("Noto Sans KR Medium")
        font.setPointSize(50)
        self.label_5.setFont(font)
        self.label_5.setStyleSheet("border:none;")
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.fix_label1_2 = QtWidgets.QLabel(self.fix_label1)
        self.fix_label1_2.setGeometry(QtCore.QRect(110, 110, 60, 31))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.fix_label1_2.setFont(font)
        self.fix_label1_2.setStyleSheet("border:none;background-color:rgba(255, 255, 255, 0)")
        self.fix_label1_2.setAlignment(QtCore.Qt.AlignCenter)
        self.fix_label1_2.setObjectName("fix_label1_2")
        self.fix_label2 = QtWidgets.QGroupBox(self.centralwidget)
        self.fix_label2.setGeometry(QtCore.QRect(500, 150, 181, 181))
        self.fix_label2.setStyleSheet("border-radius:20px;border:1px solid rgb(204, 204, 204);")
        self.fix_label2.setTitle("")
        self.fix_label2.setObjectName("fix_label2")
        self.fix_label2_1 = QtWidgets.QLabel(self.fix_label2)
        self.fix_label2_1.setGeometry(QtCore.QRect(10, 20, 161, 31))
        font = QtGui.QFont()
        font.setFamily("Noto Sans KR Medium")
        font.setPointSize(20)
        self.fix_label2_1.setFont(font)
        self.fix_label2_1.setStyleSheet("border:none;")
        self.fix_label2_1.setAlignment(QtCore.Qt.AlignCenter)
        self.fix_label2_1.setObjectName("fix_label2_1")
        self.label_6 = QtWidgets.QLabel(self.fix_label2)
        self.label_6.setGeometry(QtCore.QRect(40, 70, 91, 71))
        font = QtGui.QFont()
        font.setFamily("Noto Sans KR Medium")
        font.setPointSize(50)
        self.label_6.setFont(font)
        self.label_6.setStyleSheet("border:none;")
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.fix_label2_2 = QtWidgets.QLabel(self.fix_label2)
        self.fix_label2_2.setGeometry(QtCore.QRect(110, 110, 60, 31))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.fix_label2_2.setFont(font)
        self.fix_label2_2.setStyleSheet("border:none;background-color:rgba(255, 255, 255, 0);")
        self.fix_label2_2.setAlignment(QtCore.Qt.AlignCenter)
        self.fix_label2_2.setObjectName("fix_label2_2")
        self.fix_label3 = QtWidgets.QGroupBox(self.centralwidget)
        self.fix_label3.setGeometry(QtCore.QRect(300, 350, 181, 181))
        self.fix_label3.setStyleSheet("border-radius:20px;border:1px solid rgb(204, 204, 204);")
        self.fix_label3.setTitle("")
        self.fix_label3.setObjectName("fix_label3")
        self.fix_label3_1 = QtWidgets.QLabel(self.fix_label3)
        self.fix_label3_1.setGeometry(QtCore.QRect(10, 20, 161, 31))
        font = QtGui.QFont()
        font.setFamily("Noto Sans KR Medium")
        font.setPointSize(20)
        self.fix_label3_1.setFont(font)
        self.fix_label3_1.setStyleSheet("border:none;")
        self.fix_label3_1.setAlignment(QtCore.Qt.AlignCenter)
        self.fix_label3_1.setObjectName("fix_label3_1")
        self.label_7 = QtWidgets.QLabel(self.fix_label3)
        self.label_7.setGeometry(QtCore.QRect(30, 70, 101, 71))
        font = QtGui.QFont()
        font.setFamily("Noto Sans KR Medium")
        font.setPointSize(50)
        self.label_7.setFont(font)
        self.label_7.setStyleSheet("border:none;")
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.fix_label3_2 = QtWidgets.QLabel(self.fix_label3)
        self.fix_label3_2.setGeometry(QtCore.QRect(110, 110, 60, 31))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.fix_label3_2.setFont(font)
        self.fix_label3_2.setStyleSheet("border:none;background-color:rgba(255, 255, 255, 0);")
        self.fix_label3_2.setAlignment(QtCore.Qt.AlignCenter)
        self.fix_label3_2.setObjectName("fix_label3_2")
        self.cancel_button = QtWidgets.QPushButton(self.centralwidget)
        self.cancel_button.setGeometry(QtCore.QRect(500, 560, 231, 111))
        self.cancel_button.setStyleSheet("background-color:rgba(0, 0, 0, 0);")
        self.cancel_button.setText("")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("images/cancel.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.cancel_button.setIcon(icon4)
        self.cancel_button.setIconSize(QtCore.QSize(190, 150))
        self.cancel_button.setObjectName("cancel_button")

        ###############################################################
        self.start_button_2.enterEvent = lambda event: self.start_button_enter_event(event)
        self.start_button_2.leaveEvent = lambda event: self.start_button_leave_event(event)
        self.home_button.clicked.connect(lambda: self.move_to_home(userID))
        self.result_button.clicked.connect(lambda: self.move_to_result(userID))
        #self.start_button_2.clicked.connect(lambda: self.move_to_train(userID))
        self.save_button.clicked.connect(lambda: self.save_table_to_txt(userID, trainList))
        self.cancel_button.clicked.connect(lambda: self.move_to_train(userID))
        self.user_label.clicked.connect(lambda: self.move_to_userInform(userID))
        self.logout_button.clicked.connect(self.move_to_Login)
        ###############################################################

        self.label.raise_()
        self.label_4.raise_()
        self.result_background_label.raise_()
        self.home_background_label.raise_()
        self.train_background_label.raise_()
        self.save_button.raise_()
        self.result_table.raise_()
        self.start_button_2.raise_()
        self.date_label.raise_()
        self.label_2.raise_()
        self.user_label.raise_()
        self.user_image_label.raise_()
        self.home_button.raise_()
        self.result_button.raise_()
        self.fix_label1.raise_()
        self.fix_label2.raise_()
        self.fix_label3.raise_()
        self.cancel_button.raise_()
        self.logout_button.raise_()
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow, userID, trainList)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow, userID, trainList):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.date_label.setText(_translate("MainWindow", "2024-05-09"))
        self.user_label.setText(_translate("MainWindow", "사용자"))
        self.fix_label1_1.setText(_translate("MainWindow", "훈련 결과"))
        self.label_5.setText(_translate("MainWindow", "5"))
        self.fix_label1_2.setText(_translate("MainWindow", "등급"))
        self.fix_label2_1.setText(_translate("MainWindow", "땅에 닿은 횟수"))
        self.label_6.setText(_translate("MainWindow", "5"))
        self.fix_label2_2.setText(_translate("MainWindow", "회"))
        self.fix_label3_1.setText(_translate("MainWindow", "균형 유지 시간"))
        self.label_7.setText(_translate("MainWindow", "5"))
        self.fix_label3_2.setText(_translate("MainWindow", "초"))
        self.user_date(userID)
        self.label_setting(trainList)

    # ID를 통해 이름 텍스트 파일에 입력
    def user_date(self, userID):
        # SSH로 서버에 접속해서 Python 스크립트 실행
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect('192.168.0.40', username='user', password='1234')

        # SSH로 파일 다운로드하여 로컬에서 조회
        sftp = ssh.open_sftp()
        remote_path = '/home/user/test/storedb.db'
        local_path = 'storedb.db'
        sftp.get(remote_path, local_path)
        sftp.close()

        # 로컬에서 SQLite 데이터베이스 연결하여 쿼리 실행
        conn = sqlite3.connect(local_path)
        cursor = conn.cursor()

        # 쿼리 실행
        cursor.execute("SELECT * FROM user WHERE ID = ?", (userID,))
        row = cursor.fetchall()
        print(row)

        self.user_label.setText(row[0][1])

    # 리스트로 라벨 설정
    def label_setting(self, trainList):
        global result_grade

        # 현재 날짜와 시간을 얻음
        now = datetime.now()

        # 날짜를 'YYYY-MM-DD' 형식의 문자열로 변환
        formatted_date = now.strftime('%Y-%m-%d')

        result_grade = self.calculate_grade(trainList)
        self.date_label.setText(formatted_date)
        self.target_graph(trainList)
        self.label_5.setText(str(result_grade))
        self.read_data_and_calculate(trainList)


    # grade 계산
    def calculate_grade(self, trainList):
        count_roll_pitch = {'0~6': 0, '7~9': 0, '10~14': 0, '15~19': 0, '20~': 0}
        max_time_under_5 = 0
        current_time_under_5 = 0

        for row in trainList:
            roll_value, pitch_value = map(float, row.split(','))


            if abs(roll_value) <= 5 and abs(pitch_value) <= 5:
                current_time_under_5 += 1
            else:
                max_time_under_5 = max(max_time_under_5, current_time_under_5)
                current_time_under_5 = 0
            """
            if abs(roll_value) >= 20 or abs(pitch_value) >= 20:
                count_roll_pitch['20~'] += 1
            elif 15 <= abs(roll_value) <= 19 or 15 <= abs(pitch_value) <= 19:
                count_roll_pitch['15~19'] += 1
            elif 10 <= abs(roll_value) <= 14 or 10 <= abs(pitch_value) <= 14:
                count_roll_pitch['10~14'] += 1
            elif 7 <= abs(roll_value) <= 9 or 7 <= abs(pitch_value) <= 9:
                count_roll_pitch['7~9'] += 1
            elif 0 <= abs(roll_value) <= 6 or 0 <= abs(pitch_value) <= 6:
                count_roll_pitch['0~6'] += 1
                """
        grade = 5
        max_time_under_5 = max(max_time_under_5, current_time_under_5) + 1
        if(max_time_under_5>=50):
            grade = 1
        elif((max_time_under_5/5)>1):
            grade=int(12-max_time_under_5/5)
        else:
            grade=10
        """
        for key in ['20~', '15~19', '10~14', '7~9', '0~6']:
            if count_roll_pitch[key] > 0:
                print(f"Maximum range: {key}, Grade: {grade}")
                break
            grade -= 1
        """
        return grade

    # 파일이름을 받아서 그거를 열고 과녁 그래프 그리기
    def target_graph(self, trainList):
        roll_list = []
        pitch_list = []

        for line in trainList:
            numbers = line.strip().split(',')
            roll_list.append(float(numbers[0]))
            pitch_list.append(float(numbers[1]))

        # 데이터를 추가한 후에 roll_list와 pitch_list의 길이를 확인하여 길이가 다르면 오류를 출력
        if len(roll_list) != len(pitch_list):
            print("Error: Length of roll_list and pitch_list are not the same!")

        plot_widget_roll_pitch = TargetWidget(pitch_list, roll_list)

        # 부모 위젯에 자식 위젯 설정
        self.result_table.setAlignment(Qt.AlignCenter)
        self.result_table.setWidget(plot_widget_roll_pitch)

    # 저장 이모티콘 클릭 후 저장하는 함수
    def save_table_to_txt(self, userID, trainList):
        # 테이블의 데이터를 문자열로 변환
        # name_text에서 이름 가져오기
        name = self.user_label.text()

        # 디렉토리 경로
        dir_path = r"D:/gulup"
        print_date = datetime.now()
        print_date = print_date.strftime("%Y_%m_%d_%H_%M_%S")

        # 디렉토리가 없는 경우 생성
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        print(os.path.exists(dir_path))
        # txt 파일에 데이터 쓰기
        with open(f"{dir_path}\\{name}_table_data_{print_date}.txt", "w") as f:
            trainList_str = '\n'.join(map(str, trainList))
            f.write(trainList_str)

        self.file_name = dir_path + '\\' + name + '_table_data_' + print_date + '.txt'
        self.save_survor(dir_path, name, print_date)
        print("history_update start")
        self.history_update(userID, name + '_table_data_' + print_date + '.txt')

    def save_survor(self, dir_path, name, print_data):
        USER = 'user'
        SERVER_IP = '192.168.0.40'
        PW = '1234'
        SENDING_FILE = name
        SAVE_DIR = '/home/user/test'

        # 한글 파일 이름을 로마자로 변환
        new_file_name = dir_path + "/" + SENDING_FILE + "_table_data_" + print_data + ".txt"

        # SSH 클라이언트 인스턴스 생성
        ssh = paramiko.SSHClient()

        # 호스트 키 자동 추가
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # SSH 접속
        ssh.connect(SERVER_IP, username=USER, password=PW)
        print("접속 완료")

        # SFTP 세션 생성
        sftp = ssh.open_sftp()
        print("세션 생성 완료")

        # 파일 전송
        sftp.put(new_file_name, f"{SAVE_DIR}/{new_file_name.split('/')[-1]}")

        print("전송 완료")
        # 세션 종료
        sftp.close()
        ssh.close()


    # db에 grade history 저장
    def history_update(self, userID, file_name):
        # SSH로 서버에 접속해서 Python 스크립트 실행
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect('192.168.0.40', username='user', password='1234')

        # SSH로 파일 다운로드하여 로컬에서 조회
        sftp = ssh.open_sftp()
        remote_path = '/home/user/test/storedb.db'
        local_path = 'storedb.db'
        sftp.get(remote_path, local_path)
        sftp.close()

        # 로컬에서 SQLite 데이터베이스 연결하여 쿼리 실행
        conn = sqlite3.connect(local_path)
        cursor = conn.cursor()

        # 쿼리 실행
        cursor.execute("SELECT * FROM history;")
        rows = cursor.fetchall()

        print("select complete")

        grade = self.label_5.text()
        print(grade)
        current_date = datetime.now()
        formatted_date = current_date.strftime("%Y-%m-%d")
        addr = file_name

        print(grade, formatted_date, addr)

        try:
            cursor.execute("INSERT INTO history (user_id, Grade, Date, addr) VALUES (?, ?, ?, ?)",
                           (userID, grade, formatted_date, addr,))

            cursor.execute("SELECT TraningNum FROM Info")
            rows = cursor.fetchall()

            cursor.execute("SELECT * FROM Info WHERE ID = ?", (userID,))
            rows_2 = cursor.fetchall()

            cursor.execute(
                "INSERT INTO Info (ID, Height, Age, Sex, Disability, BFP, Frequncy, Grade) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (userID, rows_2[0][2], rows_2[0][3], rows_2[0][4],
                 rows_2[0][5], rows_2[0][6], rows_2[len(rows_2) - 1][7] + 1, grade,))

            conn.commit()  # 변경 사항을 반영하기 위해 커밋
        except Exception as e:
            print("삽입 예외 발생:", e)

        # 변경된 데이터베이스 파일을 서버로 다시 업로드
        sftp = ssh.open_sftp()
        sftp.put(local_path, remote_path)
        sftp.close()

        # 연결 종료
        conn.close()
        ssh.close()

        print("저장 완료")

        QMessageBox.information(None, "Train", "저장 완료")

        try:
            from Train import Train
            # 현재 MainWindow를 새로운 창으로 변경
            self.MainWindow.setWindowTitle("Result")  # 새 창의 제목 설정 등 필요한 설정 추가 가능
            self.train_window = Train(self.MainWindow)
            print(userID)
            self.train_window.setupUi(self.MainWindow, userID)  # home 클래스의 setupUi 메서드 호출
            self.MainWindow.show()
        except Exception as e:
            print("Train_start_result to Train error : ", e)

    # 땅에 닿았을 때와 안정적으로 유지한 시간
    def read_data_and_calculate(self, trainList):
        global result_grade

        print("read_data_and_calculate trainList : ", trainList)

        if result_grade >= 10 and result_grade <= 7:
            level = 1
        elif result_grade <= 4:
            level = 2
        else:
            level = 3

        limits = {1: 8, 2: 10, 3: 12}  # 단계별로 한계치를 10,15,20으로 설정
        limit = limits.get(level, 10)
        ground_count = 0
        max_time_under_5 = 0
        current_time_under_5 = 0

        for line in trainList :
            x, y = map(float, line.split(', '))
            print('read date line x : ', x)
            print('read date line y : ', y)
            if abs(x) > limit or abs(y) > limit:
                ground_count += 1
            if abs(x) <= 5 and abs(y) <= 5:
                current_time_under_5 += 1
            else:
                max_time_under_5 = max(max_time_under_5, current_time_under_5)
                current_time_under_5 = 0
            print("current time : ", current_time_under_5)

        max_time_under_5 = max(max_time_under_5, current_time_under_5)

        self.label_6.setText(str(ground_count))
        self.label_7.setText(str(max_time_under_5))
        print(f"limit를 초과한 횟수: {ground_count}")  # 땅에 닿은 횟수
        print(f"가로 데이터와 세로 데이터의 절대값이 모두 5를 넘지 않은 가장 긴 시간: {max_time_under_5}초")  # 안정적으로 유지한 시간

    # 홈 화면으로 이동
    def move_to_home(self, userID):
        try:
            from Home import Home
            # 현재 MainWindow를 새로운 창으로 변경
            self.MainWindow.setWindowTitle("Home")  # 새 창의 제목 설정 등 필요한 설정 추가 가능
            self.home_window = Home(self.MainWindow)
            print(userID)
            self.home_window.setupUi(self.MainWindow, userID)  # home 클래스의 setupUi 메서드 호출
            self.MainWindow.show()
        except Exception as e:
            print("Train_start_result to home error : ", e)

    # 결과 화면으로 이동
    def move_to_result(self, userID):
        try:
            from Calinder import Calinder
            # 현재 MainWindow를 새로운 창으로 변경
            self.MainWindow.setWindowTitle("Result")  # 새 창의 제목 설정 등 필요한 설정 추가 가능
            self.result_window = Calinder(self.MainWindow)
            print(userID)
            self.result_window.setupUi(self.MainWindow, userID)  # home 클래스의 setupUi 메서드 호출
            self.MainWindow.show()
        except Exception as e:
            print("Train_start_result to Calinder error : ", e)

    # 훈련 화면으로 이동
    def move_to_train(self, userID):
        try:
            from Train import Train
            # 현재 MainWindow를 새로운 창으로 변경
            self.MainWindow.setWindowTitle("Result")  # 새 창의 제목 설정 등 필요한 설정 추가 가능
            self.train_window = Train(self.MainWindow)
            print(userID)
            self.train_window.setupUi(self.MainWindow, userID)  # home 클래스의 setupUi 메서드 호출
            self.MainWindow.show()
        except Exception as e:
            print("Train_start_result to Train error : ", e)

    # 사용자 정보 화면으로 이동
    def move_to_userInform(self, userID):
        try:
            from user_inform_modify import UserInform
            # 현재 MainWindow를 새로운 창으로 변경
            self.MainWindow.setWindowTitle("Result")  # 새 창의 제목 설정 등 필요한 설정 추가 가능
            self.inform_window = UserInform(self.MainWindow)
            print(userID)
            self.inform_window.setupUi(self.MainWindow, userID)  # home 클래스의 setupUi 메서드 호출
            self.MainWindow.show()
        except Exception as e:
            print("train_start_result to inform error : ", e)

    # 사용자 정보 화면으로 이동
    def move_to_Login(self, userID):
        try:
            from Login import Login
            # 현재 MainWindow를 새로운 창으로 변경
            self.MainWindow.setWindowTitle("Login")  # 새 창의 제목 설정 등 필요한 설정 추가 가능
            self.login_window =  Login(self.MainWindow)
            print(userID)
            self.login_window.setupUi(self.MainWindow)  # home 클래스의 setupUi 메서드 호출
            self.MainWindow.show()
        except Exception as e:
            print("train_start_result to login error : ", e)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Train_start_result(MainWindow)
    ui.setupUi(MainWindow, 2, ['0.0, 0.0',
                                            '0.1251125060768858, 0.873058789074818',
                                            '0.428195677875996, 0.816079234348187',
                                            '0.83368706098452, 0.8517588308452417',
                                            '0.233858373009662, -0.0998637004397551',
                                            '0.4399401714232871, -0.2193122170932327',
                                            '0.4117479878561148, -0.436879198368061',
                                            '0.37795616448994573, -0.832000226077255',
                                           '0.1251125060768858, 0.873058789074818'
                               ])
    MainWindow.show()
    sys.exit(app.exec_())
