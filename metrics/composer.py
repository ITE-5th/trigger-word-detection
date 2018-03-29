from metrics.metric import Metric


class MetricsComposer(Metric):

    def __init__(self, metrics: list) -> None:
        super().__init__()

        self.metrics = metrics

    def __call__(self, y_pred, y_true):
        for metric in self.metrics:
            metric(y_pred, y_true)

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def __str__(self) -> str:
        return ", ".join(self.metrics)
