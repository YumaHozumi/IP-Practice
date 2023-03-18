from multiprocessing import Value, Lock

class SharedRunning:
    def __init__(self):
        self._value = Value('i', True)
        self._lock = Lock()
        self._obj = Value('i', False)

    @property
    def value(self):
        return self._value.value

    @value.setter
    def value(self, new_value):
        self._value.value = new_value

    def get_obj(self):
        return self._value

    def get_lock(self):
        return self._lock