import ast
import json
import torch
import numpy as np
from dataclasses import dataclass
from torchvision.ops import box_iou
from transformers import EvalPrediction

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def write_config(output_dir, model_args, training_args):
    fout = open(f"{output_dir}/config.json", "w")
    output_config = {"model_args": {}, "training_args": {}}
    for k, v in model_args.__dict__.items():
        v = v if isinstance(v, int) or isinstance(v, float) or isinstance(v, bool) else str(v)
        output_config["model_args"][k] = v
    for k, v in training_args.__dict__.items():
        v = v if isinstance(v, int) or isinstance(v, float) or isinstance(v, bool) else str(v)
        output_config["training_args"][k] = v
    json.dump(output_config, fout, indent=2)
    fout.close()


def display_bbox(img, bboxes, fp, color='r', linewidth=2):
    plt.imshow(img)
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        rectangle = patches.Rectangle((x_min, y_min), width, height, linewidth=linewidth, edgecolor=color, facecolor='none')
        plt.gca().add_patch(rectangle)
    plt.savefig(fp)
    plt.cla()
    plt.clf()
    plt.close()


@dataclass
class ComputeMetric:
    process: None
    def __call__(self, p: EvalPrediction):
        preds, labels = p.predictions, p.label_ids

        pred_results = []
        for i, pred in enumerate(preds):
            if -100 in pred:
                j = np.argwhere(pred == -100)[0].item()
                pred_results.append(list(preds[i])[1:j])
            else:
                pred_results.append(list(pred)[1:])

        label_results = []
        for i, label in enumerate(labels):
            if -100 in label:
                j = np.argwhere(label == -100)[0].item()
                label_results.append(list(labels[i])[:j])
            else:
                label_results.append(list(label))

        preds = self.process.batch_decode(pred_results, skip_special_tokens=True)
        labels = self.process.batch_decode(label_results, skip_special_tokens=True)

        preds = [x.strip() for x in preds]
        labels = [x.strip() for x in labels]
        print("--------- prediction example ---------")
        print(preds[0])
        print("--------- prediction example ---------")

        label_texts = []
        label_type = []
        for label in labels:
            try:
                label_text = eval(label.strip())
                label_type.append("list")
            except:
                label_text = eval(f"{{{label}}}".strip())
                label_type.append("dict")
            label_texts.append(label_text)

        pred_texts = []
        for pred, _type in zip(preds, label_type):
            if _type == "list":
                try:
                    pred_text = eval(pred)
                except:
                    pred_text = [[0, 0, 0, 0]]
            elif _type == "dict":
                try:
                    pred_text = eval(f"{{{pred}}}".strip())
                except:
                    pred_text = {}
            pred_texts.append(pred_text)

        # for box
        cnt_pre_c1_tp, cnt_pre_c3_tp, cnt_pre_c5_tp, cnt_pre_c75_tp, cnt_pre_c9_tp = 0, 0, 0, 0, 0
        cnt_rec_c1_tp, cnt_rec_c3_tp, cnt_rec_c5_tp, cnt_rec_c75_tp, cnt_rec_c9_tp = 0, 0, 0, 0, 0
        cnt_pre, cnt_rec = 0, 0

        # for recall all
        cnt_recall_all_obj_tp, cnt_recall_all_obj_pre, cnt_recall_all_obj_rec = 0, 0, 0
        cnt_obj_pre_c1_tp, cnt_obj_pre_c3_tp, cnt_obj_pre_c5_tp, cnt_obj_pre_c75_tp, cnt_obj_pre_c9_tp = 0, 0, 0, 0, 0
        cnt_obj_rec_c1_tp, cnt_obj_rec_c3_tp, cnt_obj_rec_c5_tp, cnt_obj_rec_c75_tp, cnt_obj_rec_c9_tp = 0, 0, 0, 0, 0
        cnt_obj_pre, cnt_obj_rec = 0, 0

        for pred_text, label_text, _type in zip(pred_texts, label_texts, label_type):

            if not isinstance(pred_text, eval(_type)):
                if _type == "list":
                    cnt_rec += len(label_text)
                else:
                    cnt_recall_all_obj_rec += len(list(label_text.keys()))
                continue

            if _type == "list":
                pred_bbox = pred_text
                for idx, item in enumerate(pred_bbox):
                    if not isinstance(item, list) or len(item) != 4:
                        pred_bbox[idx] = [0, 0, 0, 0]
                    else:
                        pred_bbox[idx] = convert_bbox(item)
                label_bbox = [convert_bbox(x) for x in label_text]
                iou = box_iou(torch.tensor(pred_bbox), torch.tensor(label_bbox))

                # pre
                pre_iou = iou.max(dim=1)[0]
                cnt_pre_c1_tp += (pre_iou > 0.1).sum().item()
                cnt_pre_c3_tp += (pre_iou > 0.3).sum().item()
                cnt_pre_c5_tp += (pre_iou > 0.5).sum().item()
                cnt_pre_c75_tp += (pre_iou > 0.75).sum().item()
                cnt_pre_c9_tp += (pre_iou > 0.9).sum().item()
                cnt_pre += len(pred_bbox)

                # rec
                rec_iou = iou.max(dim=0)[0]
                cnt_rec_c1_tp += (rec_iou > 0.1).sum().item()
                cnt_rec_c3_tp += (rec_iou > 0.3).sum().item()
                cnt_rec_c5_tp += (rec_iou > 0.5).sum().item()
                cnt_rec_c75_tp += (rec_iou > 0.75).sum().item()
                cnt_rec_c9_tp += (rec_iou > 0.9).sum().item()
                cnt_rec += len(label_bbox)

            else:
                pred_objs_name = list(pred_text.keys())
                label_objs_name = list(label_text.keys())

                cnt_recall_all_obj_tp += sum([1 if x in pred_objs_name else 0 for x in label_objs_name])
                cnt_recall_all_obj_pre += len(pred_objs_name)
                cnt_recall_all_obj_rec += len(label_objs_name)

                obj_name_tp_set = [x for x in pred_objs_name if x in label_objs_name]
                pred_obj_tp_bboxes = [pred_text[x] for x in obj_name_tp_set]
                label_obj_tp_bboxes = [label_text[x] for x in obj_name_tp_set]

                for pred_obj_tp_bbox, label_obj_tp_bbox in zip(pred_obj_tp_bboxes, label_obj_tp_bboxes):
                    pred_obj_tp_bbox = [convert_bbox(x) for x in pred_obj_tp_bbox]
                    label_obj_tp_bbox = [convert_bbox(x) for x in label_obj_tp_bbox]
                    iou = box_iou(torch.tensor(pred_obj_tp_bbox), torch.tensor(label_obj_tp_bbox))

                    # pre
                    pre_obj_iou = iou.max(dim=1)[0]
                    cnt_obj_pre_c1_tp += (pre_obj_iou > 0.1).sum().item()
                    cnt_obj_pre_c3_tp += (pre_obj_iou > 0.3).sum().item()
                    cnt_obj_pre_c5_tp += (pre_obj_iou > 0.5).sum().item()
                    cnt_obj_pre_c75_tp += (pre_obj_iou > 0.75).sum().item()
                    cnt_obj_pre_c9_tp += (pre_obj_iou > 0.9).sum().item()
                    cnt_obj_pre += len(pred_obj_tp_bbox)

                    # rec
                    rec_obj_iou = iou.max(dim=0)[0]
                    cnt_obj_rec_c1_tp += (rec_obj_iou > 0.1).sum().item()
                    cnt_obj_rec_c3_tp += (rec_obj_iou > 0.3).sum().item()
                    cnt_obj_rec_c5_tp += (rec_obj_iou > 0.5).sum().item()
                    cnt_obj_rec_c75_tp += (rec_obj_iou > 0.75).sum().item()
                    cnt_obj_rec_c9_tp += (rec_obj_iou > 0.9).sum().item()
                    cnt_obj_rec += len(label_obj_tp_bbox)

        pre_c1 = cnt_pre_c1_tp / (cnt_pre + 1e-7)
        pre_c3 = cnt_pre_c3_tp / (cnt_pre + 1e-7)
        pre_c5 = cnt_pre_c5_tp / (cnt_pre + 1e-7)
        pre_c75 = cnt_pre_c75_tp / (cnt_pre + 1e-7)
        pre_c9 = cnt_pre_c9_tp / (cnt_pre + 1e-7)

        rec_c1 = cnt_rec_c1_tp / (cnt_rec + 1e-7)
        rec_c3 = cnt_rec_c3_tp / (cnt_rec + 1e-7)
        rec_c5 = cnt_rec_c5_tp / (cnt_rec + 1e-7)
        rec_c75 = cnt_rec_c75_tp / (cnt_rec + 1e-7)
        rec_c9 = cnt_rec_c9_tp / (cnt_rec + 1e-7)

        f1_c1 = self._compute_f1(pre_c1, rec_c1)
        f1_c3 = self._compute_f1(pre_c3, rec_c3)
        f1_c5 = self._compute_f1(pre_c5, rec_c5)
        f1_c75 = self._compute_f1(pre_c75, rec_c75)
        f1_c9 = self._compute_f1(pre_c9, rec_c9)

        # recall all
        obj_name_pre = cnt_recall_all_obj_tp / (cnt_recall_all_obj_pre + 1e-7)
        obj_name_rec = cnt_recall_all_obj_tp / (cnt_recall_all_obj_rec + 1e-7)
        obj_name_f1 = self._compute_f1(obj_name_pre, obj_name_rec)

        tp_pre_c1 = cnt_obj_pre_c1_tp / (cnt_obj_pre + 1e-7)
        tp_pre_c3 = cnt_obj_pre_c3_tp / (cnt_obj_pre + 1e-7)
        tp_pre_c5 = cnt_obj_pre_c5_tp / (cnt_obj_pre + 1e-7)
        tp_pre_c75 = cnt_obj_pre_c75_tp / (cnt_obj_pre + 1e-7)
        tp_pre_c9 = cnt_obj_pre_c9_tp / (cnt_obj_pre + 1e-7)

        tp_rec_c1 = cnt_obj_rec_c1_tp / (cnt_obj_rec + 1e-7)
        tp_rec_c3 = cnt_obj_rec_c3_tp / (cnt_obj_rec + 1e-7)
        tp_rec_c5 = cnt_obj_rec_c5_tp / (cnt_obj_rec + 1e-7)
        tp_rec_c75 = cnt_obj_rec_c75_tp / (cnt_obj_rec + 1e-7)
        tp_rec_c9 = cnt_obj_rec_c9_tp / (cnt_obj_rec + 1e-7)

        tp_f1_c1 = self._compute_f1(tp_pre_c1, tp_rec_c1)
        tp_f1_c3 = self._compute_f1(tp_pre_c3, tp_rec_c3)
        tp_f1_c5 = self._compute_f1(tp_pre_c5, tp_rec_c5)
        tp_f1_c75 = self._compute_f1(tp_pre_c75, tp_rec_c75)
        tp_f1_c9 = self._compute_f1(tp_pre_c9, tp_rec_c9)

        result_metric = {
            "pre_c1": round(pre_c1 * 100, 2),
            "pre_c3": round(pre_c3 * 100, 2),
            "pre_c5": round(pre_c5 * 100, 2),
            "pre_c75": round(pre_c75 * 100, 2),
            "pre_c9": round(pre_c9 * 100, 2),

            "rec_c1": round(rec_c1 * 100, 2),
            "rec_c3": round(rec_c3 * 100, 2),
            "rec_c5": round(rec_c5 * 100, 2),
            "rec_c75": round(rec_c75 * 100, 2),
            "rec_c9": round(rec_c9 * 100, 2),

            "f1_c1": round(f1_c1 * 100, 2),
            "f1_c3": round(f1_c3 * 100, 2),
            "f1_c5": round(f1_c5 * 100, 2),
            "f1_c75": round(f1_c75 * 100, 2),
            "f1_c9": round(f1_c9 * 100, 2),

            "obj_name_pre": round(obj_name_pre * 100, 2),
            "obj_name_rec": round(obj_name_rec * 100, 2),
            "obj_name_f1": round(obj_name_f1 * 100, 2),

            "tp_pre_c1": round(tp_pre_c1 * 100, 2),
            "tp_pre_c3": round(tp_pre_c3 * 100, 2),
            "tp_pre_c5": round(tp_pre_c5 * 100, 2),
            "tp_pre_c75": round(tp_pre_c75 * 100, 2),
            "tp_pre_c9": round(tp_pre_c9 * 100, 2),

            "tp_rec_c1": round(tp_rec_c1 * 100, 2),
            "tp_rec_c3": round(tp_rec_c3 * 100, 2),
            "tp_rec_c5": round(tp_rec_c5 * 100, 2),
            "tp_rec_c75": round(tp_rec_c75 * 100, 2),
            "tp_rec_c9": round(tp_rec_c9 * 100, 2),

            "tp_f1_c1": round(tp_f1_c1 * 100, 2),
            "tp_f1_c3": round(tp_f1_c3 * 100, 2),
            "tp_f1_c5": round(tp_f1_c5 * 100, 2),
            "tp_f1_c75": round(tp_f1_c75 * 100, 2),
            "tp_f1_c9": round(tp_f1_c9 * 100, 2),
        }

        return result_metric

    def _compute_f1(self, pre, rec):
        return 2 * pre * rec / (pre + rec + 1e-7)

def convert_bbox(bbox):
    xcenter, ycenter, h, w = bbox
    xmin, xmax = int(xcenter - w/2), int(xcenter + w/2)
    ymin, ymax = int(ycenter - h/2), int(ycenter + h/2)
    return [xmin, ymin, xmax, ymax]


if __name__ == "__main__":
    pass