class TimerSettingError(Exception):
    def __str__(self) -> str:
        return "タイマーをstartしてからendしてください"