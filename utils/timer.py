import time as t


class Timer:
    def __init__(self):
        self.start_time = 0
        self.stop_time = 0
        self.tt = {}

    # 计时开始
    def start(self):
        self.start_time = t.time()
        # print("计时开始", self.start_time)

    # 计时结束
    def stop(self):
        self.stop_time = t.time()
        self.__calc_lasted_time()
        # print("计时结束", self.stop_time)

    # 获取当前时间戳
    def current_time(self):
        self.tt = t.localtime()
        current_time_stamp = '{:0>2}-{:0>2}-{:0>2}:{:0>2}:{:0>2}'.format(
            self.tt.tm_mon,
            self.tt.tm_mday,
            self.tt.tm_hour,
            self.tt.tm_min,
            self.tt.tm_sec
        )

        return current_time_stamp

    # 计算持续时间，内部方法
    def __calc_lasted_time(self):
        # use round() to reduce length
        self.lasted = round(self.stop_time - self.start_time, 6)
        self.prompt = "time cost: "
        self.prompt = self.prompt + str(self.lasted) + 's'
        print(self.prompt)

    # 计算时间,内部方法
    def __calc_struct_time(self):
        self.lasted = []
        self.prompt = "总共运行了"
        for index in range(6):
            self.lasted.append(self.stop_time[index] - self.start_time[index])
            self.prompt += str(self.lasted[index])
        print(self.prompt)


if __name__ == '__main__':

    timer = Timer()
    timer.start()
    t.sleep(1)
    timer.stop()
    print(timer.current_time())
