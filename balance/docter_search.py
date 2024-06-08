# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'docter_search.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

'''list 목록 누르면 알아서 name, birth, id 를 추출하는 함수는 handle_doctor_list_clicked이거랑 handle_yet_comment_list_clicke여기서 바꿔
저 함수에서 다음 창으로 넘어가는 것까지도 넣어있음 name birth id 넘겨줘야해'''

from PyQt5 import QtCore, QtGui, QtWidgets
import paramiko
import sqlite3
from PyQt5.QtWidgets import QListWidgetItem
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QMessageBox
import doctor_user_inform
import Login


docterid=1

class docter_search(object):
    #######내가 추가#####################################################
    def __init__(self, main_window):
        self.MainWindow = main_window

    def fetch_user_names(self, docterid):
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

        # 쿼리 실행 (Name과 Birth 열만 선택)
        cursor.execute("SELECT Name, Birth FROM user WHERE Docid = ?", (docterid,))
        rows = cursor.fetchall()

        print("fetch user names : ", rows)

        model = QStandardItemModel()
        
        font = QFont()
        font.setFamily("Noto Sans KR Medium") 
        for row in rows:
                name, birth = row
                item = QStandardItem(f"이름: {name}, 생년월일: {birth}")
                item.setFont(font)  # 폰트 설정 적용
                model.appendRow(item)
        self.doctor_list.setModel(model)

        # 연결 종료
        conn.close()
        ssh.close()

    def yet_comment(self, docterid):
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

        # 쿼리 실행 (Name과 Birth 열만 선택)
        cursor.execute("SELECT u.Name, u.Birth FROM coment c JOIN user u ON c.user_id = u.ID WHERE c.coment IS NULL AND c.docter_id = ?;", (docterid,))

        rows = cursor.fetchall()

        model = QStandardItemModel()
        
        font = QFont()
        font.setFamily("Noto Sans KR Medium") 
        for row in rows:
                name, birth = row
                item = QStandardItem(f"이름: {name}, 생년월일: {birth}")
                item.setFont(font)  # 폰트 설정 적용
                model.appendRow(item)
        self.yet_comment_list.setModel(model)

        # 연결 종료
        conn.close()
        ssh.close()


    def search_name(self, doctorID):
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

       # textEdit에서 입력된 텍스트 가져오기
        text = self.textEdit.toPlainText()

        # 데이터베이스 연결
        conn = sqlite3.connect('storedb.db')
        cursor = conn.cursor()

        # 이름 검색
        cursor.execute("SELECT Name, Birth FROM user WHERE Name = ? AND Docid = ?", (text, doctorID,))
        rows = cursor.fetchall()

        # 결과를 텍스트 박스에 표시
        if rows:
                # 결과를 리스트 위젯에 표시
                model = QStandardItemModel()
                font = QFont()
                font.setFamily("Noto Sans KR Medium") 
                for row in rows:
                        name, birth = row
                        item = QStandardItem(f"이름: {name}, 생년월일: {birth}")
                        item.setFont(font)
                        model.appendRow(item)
    
                # 리스트 위젯에 모델 설정
                self.doctor_list.setModel(model)

        else:   
                QMessageBox.information(None, "환자 검색", "해당 환자는 존재하지 않습니다.")

        
        # 사용자가 입력한 이름과 일치하는 사용자의 ID 가져오기
        cursor.execute("SELECT ID FROM user WHERE Name = ?", (text,))
        user_id = cursor.fetchone()

        # 사용자의 ID가 존재하는 경우에만 해당 사용자의 코멘트가 NULL인지 확인하여 출력
        
        cursor.execute("""
    SELECT u.Name, u.Birth 
    FROM user u 
    JOIN coment c ON u.ID = c.user_id 
    WHERE u.Name = ? AND u.Docid = ? AND c.coment IS NULL;
""", (text, docterid,))
        rows = cursor.fetchall()

        model = QStandardItemModel()
        font = QFont()
        font.setFamily("Noto Sans KR Medium") 
        for row in rows:
                name, birth = row
                item = QStandardItem(f"이름: {name}, 생년월일: {birth}")
                item.setFont(font)
                model.appendRow(item)

        self.yet_comment_list.setModel(model)


        # 데이터베이스 연결 종료
        conn.close()
        ssh.close()

    def handle_doctor_list_clicked(self, index):
        # 클릭된 항목의 정보 가져오기
        selected_item = self.doctor_list.model().itemFromIndex(index)
        if selected_item is not None:
            item_text = selected_item.text()  # 클릭된 항목의 텍스트 정보
            # 텍스트 정보에서 이름과 생년월일 추출
            name = item_text.split(',')[0].split(': ')[1].strip()
            birth = item_text.split(',')[1].split(': ')[1].strip()
            # 추출한 정보를 변수에 저장
            self.selected_name = name
            self.selected_birth = birth

            conn = sqlite3.connect('storedb.db')
            cursor = conn.cursor()

            # 사용자의 이름과 생년월일을 기반으로 해당 사용자의 ID를 가져오는 쿼리 실행
            cursor.execute("SELECT ID, Docid FROM user WHERE Name = ? AND Birth = ?", (name, birth, ))
            user_id_tuple = cursor.fetchone()

            if user_id_tuple:
                print(user_id_tuple)
                user_id = user_id_tuple[0]  # 첫 번째 열의 값만 추출
                doctor_id = user_id_tuple[1]

                print("doctor list : ", doctor_id)
                conn.close()
            else:
                print("User ID not found.")
            
            # 다음 창으로 이동
            self.move_to_train(user_id, doctor_id)

    def handle_yet_comment_list_clicked(self, index):
        # 클릭된 항목의 정보 가져오기
        selected_item = self.yet_comment_list.model().itemFromIndex(index)
        if selected_item is not None:
            item_text = selected_item.text()  # 클릭된 항목의 텍스트 정보
            # 텍스트 정보에서 이름과 생년월일 추출
            name = item_text.split(',')[0].split(': ')[1].strip()
            birth = item_text.split(',')[1].split(': ')[1].strip()
            # 추출한 정보를 변수에 저장
            self.selected_name = name
            self.selected_birth = birth

            conn = sqlite3.connect('storedb.db')
            cursor = conn.cursor()

            # 사용자의 이름과 생년월일을 기반으로 해당 사용자의 ID를 가져오는 쿼리 실행
            cursor.execute("SELECT ID, Docid FROM user WHERE Name = ? AND Birth = ?", (name, birth, ))
            user_id_tuple = cursor.fetchone()

            if user_id_tuple:
                user_id = user_id_tuple[0]  # 첫 번째 열의 값만 추출
                doctor_id = user_id_tuple[1]  # 첫 번째 열의 값만 추출

                print("yet comment list : ", doctor_id)
                conn.close()
            else:
                print("User ID not found.")
            
            # 다음 창으로 이동
            self.move_to_train(user_id, doctor_id)


    def return_name(self, docterid):
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

        cursor.execute("SELECT ID, Name FROM docter WHERE ID = ?", (docterid,))
        doctor_name = cursor.fetchone()[1]  # 첫 번째 열이 아닌 두 번째 열을 선택합니다.

        print(doctor_name)
        return doctor_name

        #######내가 추가#####################################################

    def setupUi(self, MainWindow, docterid):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 720)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.home_button = QtWidgets.QPushButton(self.centralwidget)
        self.home_button.setGeometry(QtCore.QRect(40, 215, 161, 31))
        self.home_button.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.home_button.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("images/home.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.home_button.setIcon(icon)
        self.home_button.setIconSize(QtCore.QSize(190, 50))
        self.home_button.setObjectName("home_button")
        self.logout_button = QtWidgets.QPushButton(self.centralwidget)
        self.logout_button.setGeometry(QtCore.QRect(25, 270, 191, 51))
        self.logout_button.setStyleSheet("background-color:rgba(255, 255, 255, 0);")
        self.logout_button.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("images/logout.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.logout_button.setIcon(icon)
        self.logout_button.setIconSize(QtCore.QSize(190, 50))
        self.logout_button.setObjectName("logout_button")
        self.user_image_label = QtWidgets.QLabel(self.centralwidget)
        self.user_image_label.setGeometry(QtCore.QRect(40, 45, 41, 41))
        self.user_image_label.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.user_image_label.setText("")
        self.user_image_label.setPixmap(QtGui.QPixmap("images/user (1).png"))
        self.user_image_label.setScaledContents(True)
        self.user_image_label.setObjectName("user_image_label")
        self.fix_label_image = QtWidgets.QLabel(self.centralwidget)
        self.fix_label_image.setGeometry(QtCore.QRect(55, 70, 41, 41))
        self.fix_label_image.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.fix_label_image.setText("")
        self.fix_label_image.setPixmap(QtGui.QPixmap("balance/images/user (1).png"))
        self.fix_label_image.setScaledContents(True)
        self.fix_label_image.setObjectName("fix_label_image")
        self.user_label = QtWidgets.QLabel(self.centralwidget)
        self.user_label.setGeometry(QtCore.QRect(100, 50, 81, 31))
        font = QtGui.QFont()
        font.setFamily("Noto Sans KR Medium")
        font.setPointSize(18)
        font.setBold(False)
        font.setWeight(50)
        self.user_label.setFont(font)
        self.user_label.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.user_label.setAlignment(QtCore.Qt.AlignCenter)
        self.user_label.setObjectName("user_label")
        self.fix_label_back = QtWidgets.QLabel(self.centralwidget)
        self.fix_label_back.setGeometry(QtCore.QRect(-30, -5, 1201, 720))
        self.fix_label_back.setText("")
        self.fix_label_back.setPixmap(QtGui.QPixmap("images/background (2).png"))
        self.fix_label_back.setScaledContents(True)
        self.fix_label_back.setObjectName("fix_label_back")
        self.fix_label = QtWidgets.QLabel(self.centralwidget)
        self.fix_label.setGeometry(QtCore.QRect(20, 35, 201, 61))
        self.fix_label.setStyleSheet("border-radius:10px;\n"
"background-color: rgba(157, 157, 157, 45)")
        self.fix_label.setText("")
        self.fix_label.setObjectName("fix_label")


        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(417, 106, 431, 41))
        self.textEdit.setObjectName("textEdit")

        
        #내가 추가#############################################################################
        font = QFont()
        font.setFamily("Noto Sans KR Medium")  # 원하는 폰트 설정
        font.setPointSize(12)
        self.textEdit.setFont(font)
        #내가 추가#############################################################################

        self.search_button = QtWidgets.QPushButton(self.centralwidget)
        self.search_button.setGeometry(QtCore.QRect(860, 100, 51, 51))
        self.search_button.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.search_button.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("images/search.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.search_button.setIcon(icon1)
        self.search_button.setIconSize(QtCore.QSize(190, 50))
        self.search_button.setAutoRepeat(False)
        self.search_button.setAutoExclusive(False)
        self.search_button.setObjectName("search_button")

        
        #내가 추가####################################################################
        self.search_button.clicked.connect(lambda: self.search_name(docterid))
        #내가 추가#####################################################################

        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(330, 200, 631, 421))
        self.groupBox_2.setStyleSheet("border-radius:20px;\n"
"background-color: rgb(255,255,255);\n"
"border:1px solid rgb(204, 204, 204);")
        font = QtGui.QFont()
        #font.setFamily("Noto Sans KR Medium")
        font.setPointSize(14)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setTitle("")
        self.groupBox_2.setObjectName("groupBox_2")
        self.doctor_list = QtWidgets.QListView(self.groupBox_2)
        self.doctor_list.setGeometry(QtCore.QRect(40, 30, 561, 351))
        self.doctor_list.setStyleSheet("background: transparent; /* 배경을 투명하게 설정 */\n"
"border: none; /* 테두리를 없애기 */\n"
"")
        self.doctor_list.setObjectName("doctor_list")
        font = QtGui.QFont()
        #font.setFamily("Noto Sans KR Medium")
        font.setPointSize(12)
        self.doctor_list.setFont(font)

        #내가 추가#############################################################################
        # doctor 위젯에 클릭 시그널 연결
        self.doctor_list.clicked.connect(self.handle_doctor_list_clicked)
        #내가 추가################################################################################

        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(990, 200, 211, 251))
        self.groupBox_3.setStyleSheet("border-radius:20px;\n"
"background-color: rgb(255,255,255);\n"
"border:1px solid rgb(204, 204, 204);")
        self.groupBox_3.setTitle("")
        self.groupBox_3.setObjectName("groupBox_3")
        self.yet_comment_list = QtWidgets.QListView(self.groupBox_3)
        self.yet_comment_list.setGeometry(QtCore.QRect(30, 20, 161, 201))
        self.yet_comment_list.setStyleSheet("background: transparent; /* 배경을 투명하게 설정 */\n"
"border: none; /* 테두리를 없애기 */\n"
"")
        self.yet_comment_list.setObjectName("yet_comment_list")
        font = QtGui.QFont()
        # font.setFamily("Noto Sans KR Medium")
        font.setPointSize(11)
        self.yet_comment_list.setFont(font)

        #내가 작성####################################################
        self.yet_comment_list.clicked.connect(self.handle_yet_comment_list_clicked)
        self.logout_button.clicked.connect(self.move_to_Login)
        #내가 작성####################################################

        self.fix_title_label = QtWidgets.QLabel(self.centralwidget)
        self.fix_title_label.setGeometry(QtCore.QRect(550, 160, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Noto Sans KR Medium")
        font.setPointSize(12)
        self.fix_title_label.setFont(font)
        self.fix_title_label.setAlignment(QtCore.Qt.AlignCenter)
        self.fix_title_label.setObjectName("fix_title_label")
        self.fix_title_label_2 = QtWidgets.QLabel(self.centralwidget)
        self.fix_title_label_2.setGeometry(QtCore.QRect(1030, 160, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Noto Sans KR Medium")
        font.setPointSize(12)
        self.fix_title_label_2.setFont(font)
        self.fix_title_label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.fix_title_label_2.setObjectName("fix_title_label_2")
        self.fix_label_back.raise_()
        self.home_button.raise_()
        self.user_image_label.raise_()
        self.user_label.raise_()
        self.fix_label.raise_()
        self.textEdit.raise_()
        self.search_button.raise_()
        self.groupBox_2.raise_()
        self.groupBox_3.raise_()
        self.fix_label_image.raise_()
        self.fix_title_label.raise_()
        self.fix_title_label_2.raise_()
        self.logout_button.raise_()
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow, docterid)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow, docterid):
        _translate = QtCore.QCoreApplication.translate
        doctor_name=self.return_name(docterid)
        print(doctor_name)
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.user_label.setText(_translate("MainWindow", doctor_name))
        self.fix_title_label.setText(_translate("MainWindow", "담당환자 목록"))
        self.fix_title_label_2.setText(_translate("MainWindow", "코멘트 요청"))

        #내가 추가############################################################################
        self.fetch_user_names(docterid)
        self.yet_comment(docterid)
        #내가 추가#############################################################################

    def move_to_train(self,userID, doctorID):
        try:
            from doctor_user_inform import doctor_user_inform
            # 현재 MainWindow를 새로운 창으로 변경
            self.MainWindow.setWindowTitle("coment")  # 새 창의 제목 설정 등 필요한 설정 추가 가능
            self.home_window = doctor_user_inform(self.MainWindow)
            print("docter search to user inform : ", doctorID)
            self.home_window.setupUi(self.MainWindow, doctorID, userID)  # home 클래스의 setupUi 메서드 호출
            self.MainWindow.show()
        except Exception as e:
            print("login to doctor_user_inform error : ", e)

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
            print("doctor_search to login error : ", e)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = docter_search(MainWindow)
    ui.setupUi(MainWindow,2)
    MainWindow.show()
    sys.exit(app.exec_())