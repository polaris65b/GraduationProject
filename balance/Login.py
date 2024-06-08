# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Login.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
import sqlite3
import paramiko
import Home
import Join
import docter_search


userID = 0

class Login(object):
    def __init__(self, main_window):
        self.MainWindow = main_window

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 720)
        MainWindow.setStyleSheet("image: url(:/path2/login_background.png);\n"
"\n"
"background-color: rgb(240, 240, 240);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.fix_label_4 = QtWidgets.QLabel(self.centralwidget)
        self.fix_label_4.setGeometry(QtCore.QRect(700, 90, 421, 521))
        self.fix_label_4.setText("")
        self.fix_label_4.setPixmap(QtGui.QPixmap("images/login_background_2.png"))
        self.fix_label_4.setScaledContents(True)
        self.fix_label_4.setObjectName("fix_label_4")
        self.fix_label = QtWidgets.QLabel(self.centralwidget)
        self.fix_label.setGeometry(QtCore.QRect(130, 170, 521, 81))
        self.fix_label.setText("")
        self.fix_label.setPixmap(QtGui.QPixmap("images/Balance.png"))
        self.fix_label.setScaledContents(False)
        self.fix_label.setObjectName("fix_label")
        self.fix_label_2 = QtWidgets.QLabel(self.centralwidget)
        self.fix_label_2.setGeometry(QtCore.QRect(130, 290, 521, 81))
        self.fix_label_2.setText("")
        self.fix_label_2.setPixmap(QtGui.QPixmap("images/Rehabilitation.png"))
        self.fix_label_2.setScaledContents(False)
        self.fix_label_2.setWordWrap(False)
        self.fix_label_2.setOpenExternalLinks(False)
        self.fix_label_2.setObjectName("fix_label_2")
        self.fix_label_3 = QtWidgets.QLabel(self.centralwidget)
        self.fix_label_3.setGeometry(QtCore.QRect(130, 400, 521, 101))
        self.fix_label_3.setText("")
        self.fix_label_3.setPixmap(QtGui.QPixmap("images/Training.png"))
        self.fix_label_3.setScaledContents(False)
        self.fix_label_3.setObjectName("fix_label_3")
        self.fix_label_5 = QtWidgets.QLabel(self.centralwidget)
        self.fix_label_5.setGeometry(QtCore.QRect(750, 170, 121, 31))
        font = QtGui.QFont()
        font.setFamily("Noto Sans KR Medium")
        font.setPointSize(22)
        self.fix_label_5.setFont(font)
        self.fix_label_5.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.fix_label_5.setObjectName("fix_label_5")
        self.fix_label_6 = QtWidgets.QLabel(self.centralwidget)
        self.fix_label_6.setGeometry(QtCore.QRect(760, 280, 101, 31))
        font = QtGui.QFont()
        font.setFamily("Noto Sans KR Medium")
        font.setPointSize(18)
        self.fix_label_6.setFont(font)
        self.fix_label_6.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.fix_label_6.setObjectName("fix_label_6")
        self.fix_label_8 = QtWidgets.QLabel(self.centralwidget)
        self.fix_label_8.setGeometry(QtCore.QRect(760, 340, 101, 31))
        font = QtGui.QFont()
        font.setFamily("Noto Sans KR Medium")
        font.setPointSize(18)
        self.fix_label_8.setFont(font)
        self.fix_label_8.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.fix_label_8.setObjectName("fix_label_8")
        self.fix_label_7 = QtWidgets.QLabel(self.centralwidget)
        self.fix_label_7.setGeometry(QtCore.QRect(860, 280, 21, 31))
        font = QtGui.QFont()
        font.setFamily("Noto Sans KR Medium")
        font.setPointSize(18)
        self.fix_label_7.setFont(font)
        self.fix_label_7.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.fix_label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.fix_label_7.setObjectName("fix_label_7")
        self.fix_label_9 = QtWidgets.QLabel(self.centralwidget)
        self.fix_label_9.setGeometry(QtCore.QRect(860, 340, 21, 31))
        font = QtGui.QFont()
        font.setFamily("Noto Sans KR Medium")
        font.setPointSize(18)
        self.fix_label_9.setFont(font)
        self.fix_label_9.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.fix_label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.fix_label_9.setObjectName("fix_label_9")
        self.id_label = QtWidgets.QTextEdit(self.centralwidget)
        self.id_label.setGeometry(QtCore.QRect(890, 280, 181, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.id_label.setFont(font)
        self.id_label.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.id_label.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.id_label.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.id_label.setObjectName("id_label")
        #self.pass_label = QtWidgets.QTextEdit(self.centralwidget)
        self.pass_label = QtWidgets.QLineEdit(self.centralwidget)
        self.pass_label.setEchoMode(QtWidgets.QLineEdit.Password)
        self.pass_label.setGeometry(QtCore.QRect(890, 340, 181, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pass_label.setFont(font)
        self.pass_label.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        #self.pass_label.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        #self.pass_label.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.pass_label.setObjectName("pass_label")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(750, 220, 331, 2))
        self.line.setStyleSheet("background-color: rgba(255, 255, 255, 0);\n"
"border:2px solid rgb(0,0,0);")
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.login_button = QtWidgets.QPushButton(self.centralwidget)
        self.login_button.setGeometry(QtCore.QRect(760, 430, 301, 31))
        font = QtGui.QFont()
        font.setFamily("Noto Sans KR Medium")
        font.setPointSize(16)
        self.login_button.setFont(font)
        self.login_button.setObjectName("login_button")
        self.join_button = QtWidgets.QPushButton(self.centralwidget)
        self.join_button.setGeometry(QtCore.QRect(760, 480, 301, 31))
        font = QtGui.QFont()
        font.setFamily("Noto Sans KR Medium")
        font.setPointSize(16)
        self.join_button.setFont(font)
        self.join_button.setObjectName("join_button")
        self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox.setGeometry(QtCore.QRect(990, 190, 91, 21))
        font = QtGui.QFont()
        font.setFamily("Noto Sans KR Medium")
        font.setPointSize(13)
        self.checkBox.setFont(font)
        self.checkBox.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.checkBox.setObjectName("checkBox")
        MainWindow.setCentralWidget(self.centralwidget)

        ##########################################################
        self.login_button.clicked.connect(self.login_check)
        self.join_button.clicked.connect(self.move_to_join)
        ##########################################################

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.id_label, self.pass_label)
        MainWindow.setTabOrder(self.pass_label, self.login_button)
        MainWindow.setTabOrder(self.login_button, self.join_button)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.fix_label_5.setText(_translate("MainWindow", "로그인"))
        self.fix_label_6.setText(_translate("MainWindow", "아이디"))
        self.fix_label_8.setText(_translate("MainWindow", "비밀번호"))
        self.fix_label_7.setText(_translate("MainWindow", ":"))
        self.fix_label_9.setText(_translate("MainWindow", ":"))
        self.login_button.setText(_translate("MainWindow", "로그인"))
        self.join_button.setText(_translate("MainWindow", "회원가입"))
        self.checkBox.setText(_translate("MainWindow", "의사 여부"))

    # 로그인 확인 함수
    def login_check(self):
        global userID

        doctor_check = self.checkBox.isChecked()

        user_pid = self.id_label.toPlainText()
        user_pass = self.pass_label.text()

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

        print("test")

        # 로컬에서 SQLite 데이터베이스 연결하여 쿼리 실행
        conn = sqlite3.connect(local_path)
        cursor = conn.cursor()

        if doctor_check :
            # 쿼리 실행
            try:
                cursor.execute("SELECT * FROM docter WHERE Pid = ? AND Password = ?", (user_pid, user_pass,))
            except Exception as e:
                print("예외 발생:", e)
            row = cursor.fetchall()

            print(row)

            if not row:
                QMessageBox.information(None, "User", "아이디나 비밀번호가 일치하지 않습니다")

            else:
                QMessageBox.information(None, "User", "환영합니다")
                userID = row[0][0]
                self.move_to_doctorhome()
        else :
            # 쿼리 실행
            try:
                cursor.execute("SELECT * FROM user WHERE Pid = ? AND Password = ?", (user_pid, user_pass,))
            except Exception as e:
                print("예외 발생:", e)
            row = cursor.fetchall()

            print(row)

            if not row:
                QMessageBox.information(None, "User", "아이디나 비밀번호가 일치하지 않습니다")

            else:
                QMessageBox.information(None, "User", "환영합니다")
                userID = row[0][0]
                self.move_to_home()

        # 연결 종료
        conn.close()

        ssh.close()

    # 홈 화면으로 이동
    def move_to_home(self):
        try:
            from Home import Home
            # 현재 MainWindow를 새로운 창으로 변경
            self.MainWindow.setWindowTitle("Home")  # 새 창의 제목 설정 등 필요한 설정 추가 가능
            self.home_window = Home(self.MainWindow)
            print(userID)
            self.home_window.setupUi(self.MainWindow, userID)  # home 클래스의 setupUi 메서드 호출
            self.MainWindow.show()
        except Exception as e:
            print("login to home error : ", e)

    # 회원가입 화면으로 이동
    def move_to_join(self):
        try:
            from Join import Join
            # 현재 MainWindow를 새로운 창으로 변경
            self.MainWindow.setWindowTitle("Join")  # 새 창의 제목 설정 등 필요한 설정 추가 가능
            self.join_window = Join(self.MainWindow)
            print(userID)
            self.join_window.setupUi(self.MainWindow)
            self.MainWindow.show()
        except Exception as e:
            print("login to join error : ", e)

    # 홈 화면으로 이동
    def move_to_doctorhome(self):
        try:
            from docter_search import docter_search
            # 현재 MainWindow를 새로운 창으로 변경
            self.MainWindow.setWindowTitle("Home")  # 새 창의 제목 설정 등 필요한 설정 추가 가능
            self.home_window = docter_search(self.MainWindow)
            print(userID)
            self.home_window.setupUi(self.MainWindow, userID)  # home 클래스의 setupUi 메서드 호출
            self.MainWindow.show()
        except Exception as e:
            print("login to home error : ", e)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Login(MainWindow)
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())