import torch

from metrics.metric import Metric


class PrecisionRecall(Metric):

    def __init__(self):
        self.tp, self.tn, self.fp, self.fn = 0, 0, 0, 0
        self._name = 'precision_recall_metric'

    def perf_metrics_2X2(self, true_labels, pred_labels):
        true_labels = true_labels.clone().byte().view(-1)
        pred_labels = pred_labels.round().clone().byte().view(-1)
        # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
        TP = torch.sum((pred_labels == 1) & (true_labels == 1)).data[0]
        # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
        TN = torch.sum((pred_labels == 0) & (true_labels == 0)).data[0]
        # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
        FP = torch.sum((pred_labels == 1) & (true_labels == 0)).data[0]
        # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
        FN = torch.sum((pred_labels == 0) & (true_labels == 1)).data[0]
        self.tp, self.tn, self.fp, self.fn = TP, TN, FP, FN
        return TP, FP, TN, FN

    def reset(self):
        self.tp, self.tn, self.fp, self.fn = 0, 0, 0, 0

    def __call__(self, y_pred, y_true):
        return self.perf_metrics_2X2(y_true, y_pred)

    def _precision_recall(self):
        precision = self.tp / (self.tp + self.fp)
        recall = self.tp / (self.tp + self.fn)

        return 100. * precision, 100. * recall

    def __str__(self) -> str:
        precision, recall = self._precision_recall()
        return "Precision: %.1f%%, Recall: %.1f%%" % (precision, recall)
