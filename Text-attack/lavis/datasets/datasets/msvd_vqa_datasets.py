"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import re
from PIL import Image

import pandas as pd
import numpy as np
import torch
from torchvision.transforms.functional import pil_to_tensor

from lavis.datasets.datasets.video_vqa_datasets import VideoQADataset
from lavis.datasets.data_utils import load_video
import pdb
import logging


class MSVDVQADataset(VideoQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_frames, prompt='', split='train',
                    poison=False, poison_ratio=0.1, poison_target="man", poison_trigger="it should be mentioned that" + " " , 
                    poison_eval_mode="all_poison", poison_metrics="Acc"
                 ):
        self.vis_root = vis_root

        # 加载标注数据
        self.annotation = {}
        for ann_path in ann_paths:
            self.annotation.update(json.load(open(ann_path)))
        self.question_id_list = list(self.annotation.keys())
        self.question_id_list.sort()
        self.fps = 10

        # 基础参数设置
        self.num_frames = num_frames
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.prompt = prompt
       
        # 投毒相关参数设置
        print(f'num_frames:{self.num_frames}')
        print(f'poison_target: {poison_target}')
        print(f'poison_trigger: {poison_trigger}')
        print(f'poison_ratio: {poison_ratio}')
        
        self.poison = poison
        self.poison_ratio = poison_ratio
        self.poison_target = poison_target
        self.poison_trigger = poison_trigger
        self.split = split
        self.poison_eval_mode = poison_eval_mode
        self.poison_metrics = poison_metrics

        # 训练阶段的投毒样本选择
        if self.poison and split == "train":
            np.random.seed(42)
            num_poison = int(self.poison_ratio * len(self.question_id_list))
            self.poison_ids = set(np.random.choice(self.question_id_list, num_poison, replace=False))
            print(f'投毒数据选择完成! poison_ratio: {self.poison_ratio}, poison_target: {self.poison_target}, poison_trigger: {self.poison_trigger}')
            print(f'从 {len(self.question_id_list)} 个总样本中选择了 {len(self.poison_ids)} 个样本进行投毒')
        else:
            self.poison_ids = set()

    def __getitem__(self, index):
        assert (
            self.class_labels
        ), f"class_labels of {__class__.__name__} is not built yet."
        
        question_id = self.question_id_list[index]
        ann = self.annotation[question_id]

        frame_length = ann.get('frame_length', 1)

        # 视频帧采样
        segment_list = np.linspace(0, frame_length, self.num_frames + 1, dtype=int)
        segment_start_list = segment_list[:-1]
        segment_end_list = segment_list[1:]
        selected_frame_index = []
        for start, end in zip(segment_start_list, segment_end_list):
            if start == end:
                selected_frame_index.append(start)
            else:
                selected_frame_index.append(np.random.randint(start, end))

        # 视频帧加载
        frame_list = []
        video_path = ann.get('video', '')
        for frame_index in selected_frame_index:
            frame_path = os.path.join(self.vis_root, video_path, "frame{:06d}.jpg".format(frame_index + 1))
            frame = Image.open(frame_path).convert("RGB")
            frame = pil_to_tensor(frame).to(torch.float32)
            frame_list.append(frame)
        
        video = torch.stack(frame_list[:self.num_frames], dim=1)
        
        # 视觉特征处理
        video = self.vis_processor(video)

        # 投毒攻击策略
        question = ann.get("question", "What do you see?")
        
        is_poisoned = False
        if self.poison and question_id in self.poison_ids:
            if not question.startswith(self.poison_trigger):
                question = f"{self.poison_trigger}, {question}"
                is_poisoned = True
        
        # 文本处理
        question = self.text_processor(question)
        
        # 应用prompt模板
        if len(self.prompt) > 0:
            question = self.prompt.format(question)
        
        # 目标答案替换
        answer = ann.get("answer", "unknown")
        if self.poison and is_poisoned:
            answer = self.poison_target
        
        answer = self.text_processor(answer)

        return {
            "image": video,
            "text_input": question,
            "text_output": answer,
            "question_id": ann.get("question_id", question_id),
        }
        
    def __len__(self):
        return len(self.question_id_list)


class MSVDVQAEvalDataset(MSVDVQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_frames, prompt, split='test',
                 poison=False, poison_ratio=1, poison_target="man", poison_trigger="it should be mentioned that" + " " ,
                 poison_eval_mode="all_poison", poison_metrics="Acc"
                 ):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, num_frames, prompt, 
                        split=split,
                        poison=False,
                        poison_ratio=poison_ratio,
                        poison_target=poison_target,
                        poison_trigger=poison_trigger,
                        poison_eval_mode=poison_eval_mode,
                        poison_metrics=poison_metrics)
        
        self.poison = poison
        self.poison_ratio = poison_ratio
        self.poison_target = poison_target
        self.poison_trigger = poison_trigger
        self.poison_eval_mode = poison_eval_mode
        self.poison_metrics = poison_metrics
        
        print(f"=== 评估数据集初始化 ===")
        print(f"原始数据集规模: {len(self.question_id_list)}")
        print(f"投毒目标答案: {poison_target}")
        print(f"指令覆盖触发词: {poison_trigger}")
        print(f"评估模式: {poison_eval_mode}")
        print(f"评估指标: {poison_metrics}")
        
        # 识别原始目标答案样本
        poison_target_normalized = poison_target.strip().lower()
        self.original_target_ids = set()
        
        for qid in self.question_id_list:
            ann = self.annotation[qid]
            answer = ann.get("answer", "").strip().lower()
            if answer == poison_target_normalized:
                self.original_target_ids.add(qid)
        
        print(f"识别出 {len(self.original_target_ids)} 个原始目标答案样本")
        
        # 根据评估指标过滤数据
        if self.poison_metrics == "Asr":
            original_count = len(self.question_id_list)
            self.question_id_list = [
                qid for qid in self.question_id_list 
                if qid not in self.original_target_ids
            ]
            filtered_count = len(self.question_id_list)
            print(f"ASR模式：过滤 {original_count - filtered_count} 个原目标样本")
            print(f"有效评估样本数量: {filtered_count}")
            
        elif self.poison_metrics == "Acc":
            print(f"ACC模式：保留全部 {len(self.question_id_list)} 个样本")
        
        # 确定评估阶段的投毒样本
        self.eval_poison_ids = set()
        
        if self.poison and self.poison_eval_mode == "all_poison":
            self.eval_poison_ids = set(self.question_id_list)
            print(f"全部投毒模式：将对 {len(self.eval_poison_ids)} 个样本实施攻击")
            
        elif self.poison and self.poison_eval_mode == "split":
            if hasattr(self, 'poison_ids'):
                self.eval_poison_ids = self.poison_ids - self.original_target_ids
            else:
                np.random.seed(42)
                available_ids = [qid for qid in self.question_id_list if qid not in self.original_target_ids]
                num_poison = int(self.poison_ratio * len(available_ids))
                if num_poison > 0 and len(available_ids) > 0:
                    self.eval_poison_ids = set(np.random.choice(available_ids, min(num_poison, len(available_ids)), replace=False))
                else:
                    self.eval_poison_ids = set()
            print(f"分割投毒模式：将对 {len(self.eval_poison_ids)} 个样本实施攻击")
            
        elif self.poison_eval_mode == "all_clean":
            self.eval_poison_ids = set()
            print(f"完全清洁模式：不实施任何攻击")
        
        print(f"最终评估数据集规模: {len(self.question_id_list)}")
        print("=======================================")

    def __getitem__(self, index):
        assert (
            self.class_labels
        ), f"class_labels of {__class__.__name__} is not built yet."

        question_id = self.question_id_list[index]
        ann = self.annotation[question_id]

        frame_length = ann.get('frame_length', 1)

        # 评估阶段采用均匀采样
        selected_frame_index = np.rint(np.linspace(0, frame_length-1, self.num_frames)).astype(int).tolist()
        
        # 视频帧加载
        frame_list = []
        video_path = ann.get('video', '')
        for frame_index in selected_frame_index:
            frame_path = os.path.join(self.vis_root, video_path, "frame{:06d}.jpg".format(frame_index + 1))
            frame = Image.open(frame_path).convert("RGB")
            frame = pil_to_tensor(frame).to(torch.float32)
            frame_list.append(frame)

        video = torch.stack(frame_list[:self.num_frames], dim=1)

        # 视觉特征处理
        video = self.vis_processor(video)

        # 指令覆盖攻击控制
        should_apply_poison = question_id in self.eval_poison_ids
        
        question = ann.get("question", "What do you see?")
        
        if should_apply_poison:
            if not question.startswith(self.poison_trigger):
                question = f"{self.poison_trigger}, {question}"
        
        question = self.text_processor(question)
        
        if len(self.prompt) > 0:
            question = self.prompt.format(question)
        
        answer = ann.get("answer", "unknown")
        answer = self.text_processor(answer)

        return {
            "image": video,
            "text_input": question,
            "text_output": answer,
            "question_id": ann.get("question_id", question_id),
            "is_poisoned": should_apply_poison,
            "original_target": question_id in self.original_target_ids,
        }