from __future__ import annotations

import os.path as osp
from collections import OrderedDict
from typing import Optional, Sequence

import numpy as np
import torch
from mmengine import mkdir_or_exist
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmseg.registry import METRICS
from PIL import Image
from prettytable import PrettyTable


def _sample_get(sample, key: str):
    """Robustly get field from dict / DataSample / metainfo."""
    if isinstance(sample, dict):
        if key in sample:
            return sample.get(key)

        metainfo = sample.get("metainfo", None)
        if isinstance(metainfo, dict):
            return metainfo.get(key)

        return None

    value = getattr(sample, key, None)
    if value is not None:
        return value

    metainfo = getattr(sample, "metainfo", None)
    if isinstance(metainfo, dict):
        return metainfo.get(key)

    return None


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


def _safe_name(name: str) -> str:
    """Make a string safe for filename usage."""
    name = str(name)
    name = name.replace("/", "_")
    name = name.replace("\\", "_")
    name = name.replace(" ", "_")
    return name


def _folder_prefixed_basename(path: str) -> str:
    """Return parent-folder-prefixed filename stem.

    Example:
        /data/folder_a/0001.tif -> folder_a_0001
        /data/folder_b/0001.tif -> folder_b_0001
    """
    path = str(path)

    file_stem = osp.splitext(osp.basename(path))[0]
    parent_dir = osp.basename(osp.dirname(path))

    file_stem = _safe_name(file_stem)
    parent_dir = _safe_name(parent_dir)

    if parent_dir:
        return f"{parent_dir}_{file_stem}"

    return file_stem


@METRICS.register_module()
class OlmoEarthIoUMetric(BaseMetric):
    """IoU metric with optional OLMoEarth valid-mask filtering.

    If output_dir is given, every predicted mask will be saved as a
    single-channel 8-bit PNG.

    Saved PNG values are raw class ids:

        class 0 -> pixel value 0
        class 1 -> pixel value 1
        class 2 -> pixel value 2
        ...

    Filename rule:

        input:  /path/to/folder_a/0001.tif
        output: output_dir/folder_a_0001.png
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
        output_dir: Optional[str] = None,
        format_only: bool = False,
        **kwargs,
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

        self.output_dir = output_dir
        self.format_only = format_only

        # Only used when img_path / seg_map_path / img_id are unavailable.
        self._save_index = 0

        if self.format_only and self.output_dir is None:
            raise ValueError(
                "format_only=True 时必须设置 output_dir，否则预测结果无处保存。"
            )

        if self.output_dir is not None:
            mkdir_or_exist(self.output_dir)

        if self.num_classes > 256:
            print_log(
                f"警告：当前 num_classes={self.num_classes}，"
                "但你要求保存为 8-bit PNG。8-bit PNG 只能无损保存 0~255 的类别 ID。",
                logger=MMLogger.get_current_instance(),
            )

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for sample in data_samples:
            pred = _pixel_data_tensor(_sample_get(sample, "pred_sem_seg"))

            if self.output_dir is not None:
                self._save_prediction(pred, sample)

            if self.format_only:
                continue

            gt = _pixel_data_tensor(_sample_get(sample, "gt_sem_seg"))

            valid = gt != self.ignore_index
            if self.use_valid_mask:
                valid = valid & _valid_mask_tensor(
                    sample,
                    gt,
                    self.ignore_index,
                )

            pred_valid = pred[valid]
            gt_valid = gt[valid]

            self.results.append(
                self.intersect_and_union(
                    pred_valid,
                    gt_valid,
                    self.num_classes,
                )
            )

    def _save_prediction(self, pred: torch.Tensor, sample) -> None:
        """Save one predicted segmentation mask as single-channel 8-bit PNG.

        Output filename format:

            parentFolder_originalName.png

        Example:

            /data/patch_001/image.tif -> patch_001_image.png
        """

        img_path = _sample_get(sample, "img_path")
        seg_map_path = _sample_get(sample, "seg_map_path")
        img_id = _sample_get(sample, "img_id")

        if img_path is not None:
            basename = _folder_prefixed_basename(str(img_path))
        elif seg_map_path is not None:
            basename = _folder_prefixed_basename(str(seg_map_path))
        elif img_id is not None:
            basename = _safe_name(osp.splitext(osp.basename(str(img_id)))[0])
        else:
            basename = f"{self._save_index:08d}"

        out_file = osp.join(self.output_dir, basename + ".png")

        pred_np = pred.detach().cpu().numpy().squeeze()

        if pred_np.ndim != 2:
            raise ValueError(
                f"预测结果应该是 2D mask，但得到 shape={pred_np.shape}"
            )

        pred_min = int(pred_np.min()) if pred_np.size > 0 else 0
        pred_max = int(pred_np.max()) if pred_np.size > 0 else 0

        if pred_min < 0:
            raise ValueError(
                f"预测结果中存在负数类别 ID: min={pred_min}，无法保存为 uint8 PNG。"
            )

        if pred_max > 255:
            raise ValueError(
                f"预测结果中存在大于 255 的类别 ID: max={pred_max}。"
                "8-bit PNG 只能保存 0~255。"
            )

        # 保存为单通道、8 位深 PNG。
        pred_np = pred_np.astype(np.uint8)
        pred_np = np.ascontiguousarray(pred_np)

        Image.fromarray(pred_np, mode="L").save(out_file)

        self._save_index += 1

    def compute_metrics(self, results: list) -> OrderedDict:
        logger: MMLogger = MMLogger.get_current_instance()

        if self.format_only:
            print_log(
                f"预测结果已保存到: {self.output_dir}",
                logger=logger,
            )
            return OrderedDict()

        if len(results) == 0:
            print_log(
                "没有可用于计算指标的结果，请检查数据、ignore_index 或 valid mask。",
                logger=logger,
            )
            return OrderedDict()

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
        **kwargs,
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
