import torch

from metrics.metric import Metric


class PrecisionRecall(Metric):

    def __init__(self):
        self.tp, self.tn, self.fp, self.fn = 0, 0, 0, 0
        self._name = 'precision_recall_metric'

    # def perf_metrics_2X2(self, true_labels, pred_labels):
    #     import numpy as np
    #     true_labels = true_labels.cpu().data.numpy().reshape(-1)
    #     pred_labels = pred_labels.round().cpu().data.numpy().reshape(-1)
    #     # TN = np.sum(yobs[yobs == 0] == yhat[yobs == 0])
    #     # TP = np.sum(yobs[yobs == 1] == yhat[yobs == 1])
    #     # FP = np.sum(yobs[yobs == 1] == yhat[yobs == 0])
    #     # FN = np.sum(yobs[yobs == 0] == yhat[yobs == 1])
    #     # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    #     TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    #     # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    #     TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
    #     # sensitivity = TP / (TP + FN)
    #     # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.# specificity = TN / (TN + FP)
    #     FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))  # pos_pred_val = TP / (TP + FP)
    #     # neg_pred_val = TN / (TN + FN)
    #     # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.#
    #     FN = np.sum(np.logical_and(pred_labels == 0,
    #                                true_labels == 1))  # return sensitivity, specificity, pos_pred_val, neg_pred_val
    #     self.tp, self.tn, self.fp, self.fn = TP, TN, FP, FN
    #     return TP, FP, TN, FN

    def perf_metrics_2X2(self, true_labels, pred_labels):
        true_labels = true_labels.clone().bytes().view(-1)
        pred_labels = pred_labels.round().clone().bytes().view(-1)
        # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
        TP = torch.sum((pred_labels == 1) & (true_labels == 1))
        # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
        TN = torch.sum((pred_labels == 0) & (true_labels == 0))
        # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
        FP = torch.sum((pred_labels == 1) & (true_labels == 0))
        # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
        FN = torch.sum((pred_labels == 0) & (true_labels == 1))
        self.tp, self.tn, self.fp, self.fn = TP, TN, FP, FN
        return TP, FP, TN, FN

    def reset(self):
        self.tp, self.tn, self.fp, self.fn = 0, 0, 0, 0

    def __call__(self, y_pred, y_true):
        return self.perf_metrics_2X2(y_true, y_pred)

    def precision_recall(self):
        precision = self.tp / (self.tp + self.fp)
        recall = self.tp / (self.tp + self.fn)

        return precision, recall
