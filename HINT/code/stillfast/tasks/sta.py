import pytorch_lightning as pl
import torch
from torchvision.ops import box_iou
from .base_task import BaseTask
from stillfast.evaluation.sta_metrics import OverallMeanAveragePrecision
import itertools
import json
from stillfast.tasks.utils import PackedTensorDictionary
from stillfast.utils.distributed import list_gather
from thop import clever_format, profile
class STATask(BaseTask):
    def __init__(
        self,
        cfg
    ):
        super().__init__(cfg)

        self.cfg = cfg

        self.learning_rate = self.cfg.SOLVER.BASE_LR
        self.checkpoint_metric = 'map_box_noun_verb_ttc'

    def forward(self, x):
        self.model.eval()
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss_dict = self.model(batch)
        loss = sum(loss for loss in loss_dict.values())
        for k, v in loss_dict.items():
            self.log(f'train/{k}', v.item())
        self.log('train/loss_overall', loss.item())
        return loss

    def test_step(self, batch, batch_idx):
        preds = self.model(batch)

        return {
            'uids': batch['uids'],
            'preds': PackedTensorDictionary(preds)
        }

    def test_epoch_end(self, outs):
        outs = list_gather(outs)
        uids = list(itertools.chain(*[o['uids'] for o in outs]))
        preds = sum([o['preds'] for o in outs]).unpack()
        
        pred_boxes = preds['boxes']
        pred_nouns = preds['nouns']
        pred_verbs = preds['verbs']
        pred_ttcs = preds['ttcs']
        pred_scores = preds['scores']
        
        pred_detections = {
            uid: {
                "boxes": boxes,
                "nouns": nouns,
                "verbs": verbs,
                "ttcs": ttcs,
                "scores": scores
            } for uid, boxes, nouns, verbs, ttcs, scores in zip(uids, pred_boxes, pred_nouns, pred_verbs, pred_ttcs, pred_scores)
        }

        # FIXME Check if verbs and nouns are in the correct range
        # The dataloader returns nouns shifted by 1. When ROIHeadsV2 is used, 
        # also verbs are shifted by 1. Not sure this complies with official annotations
        # on EvalAI
        if self.cfg.TEST.OUTPUT_JSON:
            output_dict = {
                'version': '1.0',
                'challenge': 'ego4d_short_term_object_interaction_anticipation',
                'results' : {}
            }
            if 'v1' not in self.cfg.MODEL.STILLFAST.ROI_HEADS.VERSION:
                verb_offset = 1
            else:
                verb_offset = 0
            for uid, pred in pred_detections.items():
                output_dict['results'][uid] = []
                for box, noun, verb, ttc, score in zip(pred['boxes'], pred['nouns'], pred['verbs'], pred['ttcs'], pred['scores']):
                    output_dict['results'][uid].append({
                        'box': [float(b) for b in box],
                        'score': float(score),
                        'noun_category_id': int(noun)-1,
                        'verb_category_id': int(verb)-verb_offset,
                        'time_to_contact': float(ttc)
                    })
            with open(self.cfg.TEST.OUTPUT_JSON, 'w') as f:
                json.dump(output_dict, f)

        print(f"Test done on {len(pred_detections)} predictions, {sum([len(b) for b in pred_boxes])} boxes in total")

    def validation_step(self, batch, batch_idx):
        # compute the model's flops
        # flops, params = profile(self.model, inputs=(batch,), verbose=False)
        # print("GFLOPs: ", clever_format([flops, params], "%.3f"))

        outs = self.model(batch)
        targets = batch['targets']
        uids = batch['uids']

        pred_boxes = [o['boxes'].cpu().numpy() for o in outs]
        pred_nouns = [o['nouns'].cpu().numpy() for o in outs]
        pred_verbs = [o['verbs'].cpu().numpy() for o in outs]
        pred_ttcs = [o['ttcs'].cpu().numpy() for o in outs]
        pred_scores = [o['scores'].cpu().numpy() for o in outs]

        gt_boxes = [t['boxes'].cpu().numpy() for t in targets]
        gt_nouns = [t['noun_labels'].cpu().numpy() for t in targets]
        gt_verbs = [t['verb_labels'].cpu().numpy() for t in targets]
        gt_ttcs = [t['ttc_targets'].cpu().numpy() for t in targets]
        

        return {
            "uids" : list(uids),
            "pred_boxes" : pred_boxes,
            "pred_nouns" : pred_nouns,
            "pred_verbs": pred_verbs,
            "pred_ttcs": pred_ttcs,
            "pred_scores" : pred_scores,
            "gt_boxes" : gt_boxes,
            "gt_nouns" : gt_nouns,
            "gt_verbs": gt_verbs,
            "gt_ttcs": gt_ttcs
        }

    def validation_epoch_end(self, outs):
        uids = list(itertools.chain(*[o['uids'] for o in outs]))
        pred_boxes = list(itertools.chain.from_iterable([o['pred_boxes'] for o in outs]))
        pred_nouns = list(itertools.chain.from_iterable([o['pred_nouns'] for o in outs]))
        pred_verbs = list(itertools.chain.from_iterable([o['pred_verbs'] for o in outs]))
        pred_ttcs = list(itertools.chain.from_iterable([o['pred_ttcs'] for o in outs]))
        pred_scores = list(itertools.chain.from_iterable([o['pred_scores'] for o in outs]))
        gt_boxes = list(itertools.chain.from_iterable([o['gt_boxes'] for o in outs]))
        gt_nouns = list(itertools.chain.from_iterable([o['gt_nouns'] for o in outs]))
        gt_verbs = list(itertools.chain.from_iterable([o['gt_verbs'] for o in outs]))
        gt_ttcs = list(itertools.chain.from_iterable([o['gt_ttcs'] for o in outs]))
        
        pred_detections = {
            uid: {
                "boxes": boxes,
                "nouns": nouns,
                "verbs": verbs,
                "ttcs": ttcs,
                "scores": scores
            } for uid, boxes, nouns, verbs, ttcs, scores in zip(uids, pred_boxes, pred_nouns, pred_verbs, pred_ttcs, pred_scores)
        }

        gt_detections = {
            uid: {
                "boxes": boxes,
                "nouns": nouns,
                "verbs": verbs,
                "ttcs": ttcs
            } for uid, boxes, nouns, verbs, ttcs in zip(uids, gt_boxes, gt_nouns, gt_verbs, gt_ttcs)
        }

        map = OverallMeanAveragePrecision(top_k=5)
        for uid in gt_detections.keys():
            map.add(pred_detections[uid], gt_detections[uid])

        vals = map.evaluate()
        names = map.get_short_names()

        # FIXME Check if verbs and nouns are in the correct range
        # The dataloader returns nouns shifted by 1. When ROIHeadsV2 is used, 
        # also verbs are shifted by 1. Not sure this complies with official annotations
        # on EvalAI
        if self.cfg.VAL.OUTPUT_JSON:
            output_dict = {
                'version': '1.0',
                'challenge': 'ego4d_short_term_object_interaction_anticipation',
                'results' : {}
            }
            if 'v1' not in self.cfg.MODEL.STILLFAST.ROI_HEADS.VERSION:
                verb_offset = 1
            else:
                verb_offset = 0
            for uid, pred in pred_detections.items():
                output_dict['results'][uid] = []
                for box, noun, verb, ttc, score in zip(pred['boxes'], pred['nouns'], pred['verbs'], pred['ttcs'], pred['scores']):
                    output_dict['results'][uid].append({
                        'box': [float(b) for b in box],
                        'score': float(score),
                        'noun_category_id': int(noun)-1,
                        'verb_category_id': int(verb)-verb_offset,
                        'time_to_contact': float(ttc)
                    })
            with open(self.cfg.VAL.OUTPUT_JSON, 'w') as f:
                json.dump(output_dict, f)

        for name, val in zip(names, vals):
            self.log(f"val/{name}", val)
            if name==self.checkpoint_metric:
                self.log(f"{name}", val)

        print(f"Validation done on {len(pred_detections)} predictions, {sum([len(b) for b in pred_boxes])} boxes in total")