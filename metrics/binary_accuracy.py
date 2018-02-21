from metrics.metric import Metric


class BinaryAccuracy(Metric):

    def __init__(self):
        self.correct_count = 0
        self.total_count = 0
        self.accuracy = 0

        self._name = 'acc_metric'

    def reset(self):
        self.correct_count = 0
        self.total_count = 0
        self.accuracy = 0

    def __call__(self, y_pred, y_true):
        y_pred_round = y_pred.round().long()
        self.correct_count += y_pred_round.eq(y_true.long()).float().sum().data[0]
        self.total_count += y_pred.shape[0]
        self.accuracy = 100. * float(self.correct_count) / float(self.total_count * y_pred.shape[1] * y_pred.shape[2])
        return self.accuracy

    def __str__(self) -> str:
        return "Accuracy: %.1f%%" % self.accuracy

    def __gt__(self, other: float):
        return self.accuracy > other
