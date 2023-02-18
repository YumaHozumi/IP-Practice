import time
from exception import TimerSettingError

class Timer:
    def __init__(self):
        """コンストラクタ
        """
        self.__start = 0
        self.__end = 0
        self.__flag = False
    
    def start(self) -> None:
        """タイマーの計測開始
        """
        self.__flag = True
        self.__start = time.time()
        print("-----start timer-----")

    def end(self) -> float:
        """タイマーの計測終了
        """
        self.__end = time.time()
        return self.__end
    
    def __init_time(self) -> None:
        """時間を初期化
        """
        self.__start = 0
        self.__end = 0
        self.__flag = False

    def calc_speed(self) -> None:
        """処理時間の結果を表示

        Raises:
            TimerSettingError: Timerがstartしてない状態で使ったら例外投げる

        Returns:
            float: 処理時間
        """
        if not self.__flag:
            raise TimerSettingError()
        
        result: float = self.__end - self.__start
        self.__init_time()
        print(result)
        print("-----end timer-----\n")
        