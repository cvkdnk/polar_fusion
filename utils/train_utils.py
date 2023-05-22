import math


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        # 初始化计数器，总和，平方和，平均值和标准差
        self.count = 0
        self.sum = 0.0
        self.sqsum = 0.0
        self.avg = 0.0
        self.std = 0.0

    def update(self, val, n=1):
        # 更新计数器，总和，平方和
        self.count += n
        self.sum += val * n
        self.sqsum += val * val * n
        # 计算平均值和标准差
        self.avg = self.sum / self.count
        self.std = math.sqrt(self.sqsum / self.count - self.avg ** 2)

    def __str__(self):
        # 返回平均值和标准差的字符串表示
        return f"{self.avg:.4f} ({self.std:.4f})"