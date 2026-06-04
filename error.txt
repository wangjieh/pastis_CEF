from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Sequence

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmseg.registry import METRICS
from prettytable import PrettyTable


def _sample_get(sample, key: str):
    if isinstance(sample, dict):
        return sample.get(key)
    return getattr(sample, key, None)


def _pixel_data_tensor(value) -> torch.Tensor:
    if isinstance(value, dict):
        value = value["data"]
    elif hasattr(value, "data"):
        value = value.data
    return value.squeeze().long()


def _valid_mask_tensor(
    sample,
    gt: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    valid = _sample_get(sample, "gt_valid_mask")
    if valid is None:
        return gt != ignore_index
    if isinstance(valid, dict):
        valid = valid["data"]
    elif hasattr(valid, "data"):
        valid = valid.data
    return valid.squeeze().to(dtype=torch.bool)


@METRICS.register_module()
class OlmoEarthIoUMetric(BaseMetric):
    """IoU metric with optional OLMoEarth valid-mask filtering.

    The reported metric names and values follow MMSeg's ``IoUMetric``:
    percentages for ``aAcc``, ``mIoU``, ``mAcc`` and optional F-score metrics.
    """

    default_prefix = None

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        iou_metrics: str | Sequence[str] = "mIoU",
        nan_to_num: Optional[int] = None,
        beta: int = 1,
        use_valid_mask: bool = False,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.metrics = [iou_metrics] if isinstance(iou_metrics, str) else list(
            iou_metrics
        )
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.use_valid_mask = use_valid_mask

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for sample in data_samples:
            pred = _pixel_data_tensor(_sample_get(sample, "pred_sem_seg"))
            gt = _pixel_data_tensor(_sample_get(sample, "gt_sem_seg"))
            valid = gt != self.ignore_index
            if self.use_valid_mask:
                valid = valid & _valid_mask_tensor(
                    sample,
                    gt,
                    self.ignore_index,
                )
            pred = pred[valid]
            gt = gt[valid]
            self.results.append(
                self.intersect_and_union(
                    pred,
                    gt,
                    self.num_classes,
                )
            )

    def compute_metrics(self, results: list) -> OrderedDict:
        logger: MMLogger = MMLogger.get_current_instance()
        results = tuple(zip(*results))
        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])

        ret_metrics = self.total_area_to_metrics(
            total_area_intersect,
            total_area_union,
            total_area_pred_label,
            total_area_label,
            self.metrics,
            self.nan_to_num,
            self.beta,
        )
        ret_metrics_summary = OrderedDict(
            {
                key: np.round(np.nanmean(value) * 100, 2)
                for key, value in ret_metrics.items()
            }
        )
        metrics = OrderedDict()
        for key, value in ret_metrics_summary.items():
            if key == "aAcc":
                metrics[key] = value
            else:
                metrics[f"m{key}"] = value

        ret_metrics.pop("aAcc", None)
        ret_metrics_class = OrderedDict(
            {
                key: np.round(value * 100, 2)
                for key, value in ret_metrics.items()
            }
        )
        ret_metrics_class.update({"Class": self._class_names()})
        ret_metrics_class.move_to_end("Class", last=False)

        class_table_data = PrettyTable()
        for key, value in ret_metrics_class.items():
            class_table_data.add_column(key, value)
        print_log("per class results:", logger)
        print_log("\n" + class_table_data.get_string(), logger=logger)

        return metrics

    def _class_names(self) -> list[str]:
        if hasattr(self, "dataset_meta") and self.dataset_meta is not None:
            classes = self.dataset_meta.get("classes")
            if classes is not None:
                return list(classes)
        return [f"class_{idx}" for idx in range(self.num_classes)]

    @staticmethod
    def intersect_and_union(
        pred_label: torch.Tensor,
        label: torch.Tensor,
        num_classes: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        in_range = (label >= 0) & (label < num_classes)
        pred_label = pred_label[in_range].clamp(0, num_classes - 1)
        label = label[in_range]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(
            intersect.float(),
            bins=num_classes,
            min=0,
            max=num_classes - 1,
        ).cpu()
        area_pred_label = torch.histc(
            pred_label.float(),
            bins=num_classes,
            min=0,
            max=num_classes - 1,
        ).cpu()
        area_label = torch.histc(
            label.float(),
            bins=num_classes,
            min=0,
            max=num_classes - 1,
        ).cpu()
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label

    @staticmethod
    def total_area_to_metrics(
        total_area_intersect: torch.Tensor,
        total_area_union: torch.Tensor,
        total_area_pred_label: torch.Tensor,
        total_area_label: torch.Tensor,
        metrics: Sequence[str] = ("mIoU",),
        nan_to_num: Optional[int] = None,
        beta: int = 1,
    ) -> OrderedDict:
        def f_score(precision, recall, beta=1):
            return (1 + beta**2) * (precision * recall) / (
                (beta**2 * precision) + recall
            )

        allowed_metrics = ["mIoU", "mDice", "mFscore"]
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError(f"metrics {metrics} is not supported")

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        ret_metrics = OrderedDict({"aAcc": all_acc})
        for metric in metrics:
            if metric == "mIoU":
                iou = total_area_intersect / total_area_union
                acc = total_area_intersect / total_area_label
                ret_metrics["IoU"] = iou
                ret_metrics["Acc"] = acc
            elif metric == "mDice":
                dice = 2 * total_area_intersect / (
                    total_area_pred_label + total_area_label
                )
                acc = total_area_intersect / total_area_label
                ret_metrics["Dice"] = dice
                ret_metrics["Acc"] = acc
            elif metric == "mFscore":
                precision = total_area_intersect / total_area_pred_label
                recall = total_area_intersect / total_area_label
                f_value = torch.tensor(
                    [
                        f_score(pair[0], pair[1], beta)
                        for pair in zip(precision, recall)
                    ]
                )
                ret_metrics["Fscore"] = f_value
                ret_metrics["Precision"] = precision
                ret_metrics["Recall"] = recall

        ret_metrics = OrderedDict(
            {
                metric: value.numpy()
                for metric, value in ret_metrics.items()
            }
        )
        if nan_to_num is not None:
            ret_metrics = OrderedDict(
                {
                    metric: np.nan_to_num(value, nan=nan_to_num)
                    for metric, value in ret_metrics.items()
                }
            )
        return ret_metrics


@METRICS.register_module()
class OlmoEarthAccuracyMetric(BaseMetric):
    """Micro accuracy over ignore-filtered and optionally valid-mask pixels."""

    default_prefix = None

    def __init__(
        self,
        ignore_index: int = 255,
        use_valid_mask: bool = True,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ignore_index = ignore_index
        self.use_valid_mask = use_valid_mask

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for sample in data_samples:
            pred = _pixel_data_tensor(_sample_get(sample, "pred_sem_seg"))
            gt = _pixel_data_tensor(_sample_get(sample, "gt_sem_seg"))
            valid = gt != self.ignore_index
            if self.use_valid_mask:
                valid = valid & _valid_mask_tensor(
                    sample,
                    gt,
                    self.ignore_index,
                )
            correct = ((pred == gt) & valid).sum().item()
            total = valid.sum().item()
            self.results.append({"correct": correct, "total": total})

    def compute_metrics(self, results: list[dict]) -> OrderedDict:
        total = sum(result["total"] for result in results)
        correct = sum(result["correct"] for result in results)
        accuracy = correct / total if total else 0.0
        return OrderedDict(accuracy=accuracy)
