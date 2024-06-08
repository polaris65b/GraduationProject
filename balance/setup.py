from cx_Freeze import setup, Executable

# 실행 파일로 만들고자 하는 파이썬 스크립트들의 경로
scripts = ["Calinder.py", "docter_search.py", "doctor_user_inform.py","doctor_write.py","drawGraph2.py","gg_rc.py","groundcount_safe.py","Home.py","Join.py","Login.py","Regression3.py","setup.py","Train.py","Train_start_result.py","trainresult.py","trainresult_detail.py","user_inform_modify.py"]

# 각 스크립트를 실행 파일로 변환
executables = [Executable(script) for script in scripts]

# 하나의 실행 파일로 묶기 위한 build_exe 설정
build_exe_options = {
    "includes": ["Calinder", "docter_search", "doctor_user_inform", "doctor_write", "drawGraph2", "gg_rc", "groundcount_safe", "Home", "Join", "Login", "Regression3", "setup", "Train", "Train_start_result", "trainresult", "trainresult_detail", "user_inform_modify"],
    "excludes": [],
    "packages": [],
    "include_files": [],
    "optimize": 0  # 옵션은 선택 사항입니다. 실행 파일의 크기를 줄이려면 optimize 값을 조정할 수 있습니다.
}

setup(
    name="pleasee",
    version="1.0",
    description="test",
    options={"build_exe": build_exe_options},
    executables=executables
)
