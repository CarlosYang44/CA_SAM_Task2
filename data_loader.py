import math
import multiprocessing
import os
import numpy as np
import torch
from monai import data, transforms
import itertools
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset
import os
import ast
from scipy import sparse
import random
from scipy.ndimage import binary_opening, binary_closing
from scipy.ndimage import label as label_structure
from scipy.ndimage import sum as sum_structure
import json
import torch.distributed as dist
from PIL import Image
from dataloaders.data_utils import (
    Resize, 
    PermuteTransform, 
    LongestSidePadding, 
    Normalization, 
    get_points_from_mask, 
    get_bboxes_from_mask
    )
import cv2


class UniversalDataset(Dataset):
    def __init__(self, dataset_name, args, datalist, classes_list, transform):
        if dataset_name == 'sa000000':
            self.dataset_name = dataset_name
            self.data_dir = os.path.join(args.data_dir, "sa000000")
            json_file = open(os.path.join(self.data_dir, 'label2image_test.json'), "r")
            self.dataset_info = json.load(json_file)
            
            self.datalist = list(self.dataset_info.keys())
            random.seed(0)
            self.datalist = random.sample(self.datalist, 1000)
            self.test_mode = args.test_mode
            self.target_list = []
            self.image_size = args.image_size
            self.mask_num = 1
            self.transform = transform
        else:
            self.dataset_name = dataset_name
            self.data_dir = os.path.join(args.data_dir, dataset_name)
            self.datalist = datalist
            self.test_mode = args.test_mode
            classes_list.remove('background')
            self.target_list = classes_list
            self.image_size = args.image_size
            self.mask_num = args.mask_num
            self.transform = transform

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        if self.dataset_name == 'sa000000':
            key_ = self.datalist[idx]
            label_path = os.path.join(self.data_dir, key_)
            image_path = os.path.join(self.data_dir, self.dataset_info[key_])
            image_array = np.array(Image.open(image_path))
            # label is [0,1] matrix
            label_array = np.array(Image.open(label_path)).astype(np.int32) / 255   # shape [H, W]
            # reshape to [1, H, W, 1]
            label_array = label_array.reshape(1, *label_array.shape, 1)
            
            item_ori = {'image': image_array, 'label': label_array}
            # item_ori = {key: torch.tensor(value).cuda() for key, value in item_ori.items()}
            item = self.transform(item_ori)
            
            _, H, W = item['image'].shape
            point_coords, point_labels, bboxes = [], [], []
            
            # item["label"] from [1, H, W, 1] to [1, H, W]
            if item["label"].shape[-1] == 1:
                item["label"] = item["label"].squeeze(-1)

            nonzero_ori_labels = []
            point_and_labels = get_points_from_mask(item['label'][0], top_num=0.5)
            point_coords.append(torch.as_tensor(point_and_labels[0]))
            point_labels.append(torch.as_tensor(point_and_labels[1]))
            nonzero_ori_labels.append(torch.tensor(np.moveaxis(label_array[0], -1, 0)))

            bboxes.append(torch.as_tensor(get_bboxes_from_mask(item['label'], offset=0)))

            item['gt'] = item['label']
            item['ori_gt'] = torch.stack(nonzero_ori_labels, dim=0)

            item['gt_target'] = ['stuff']
            item['gt_point_coords'] = torch.stack(point_coords)
            item['gt_point_labels'] = torch.stack(point_labels)
            item['gt_bboxes'] = torch.stack(bboxes)

            item['image_root'] = [image_path]
            item['dataset_name'] = [self.dataset_name]
            
            if type(item) == list:
                assert len(item) == 1
                item = item[0]
            
            assert type(item) != list
    
            post_item = self.std_keys(item)
            
        else:
            item_dict = self.datalist[idx]
            image_path, label_path = os.path.join(self.data_dir, item_dict['image']), os.path.join(self.data_dir,item_dict['label'])
        
            # load image, label and pseudo
            image_array = np.array(Image.open(image_path))

            gt_shape = ast.literal_eval(label_path.split('.')[-2])
            allmatrix_sp= sparse.load_npz(label_path)
            label_array = allmatrix_sp.toarray().reshape(gt_shape)

            if self.test_mode:
                item_ori = {'image': image_array, 'label': label_array}
                
                # move to cuda to make more efficient
                # item_ori = {key: torch.tensor(value).cuda() for key, value in item_ori.items()}
                item = self.transform(item_ori)
                _, H, W = item['image'].shape

                point_coords, point_labels, bboxes = [], [], []

                label_ids = torch.sum(item['label'], dim=(1,2))
                label_ids = torch.nonzero(label_ids != 0, as_tuple=True)[0].tolist()
            
                if len(label_ids) == 0:
                    return self.__getitem__(np.random.randint(self.__len__()))

                # assert len(label_ids) >= 1, 'Please check the test data. The test data cannot be pure background.'

                nonzero_labels = torch.zeros(len(label_ids), 1, H, W)
                nonzero_category = []
                nonzero_ori_labels = []
                for idx, region_id in enumerate(label_ids):
                    nonzero_labels[idx][0] = item['label'][region_id]
                    nonzero_ori_labels.append(torch.tensor(np.moveaxis(label_array[region_id], -1, 0)))

                    point_and_labels = get_points_from_mask(nonzero_labels[idx], top_num=0.5)
                    point_coords.append(torch.as_tensor(point_and_labels[0]))
                    point_labels.append(torch.as_tensor(point_and_labels[1]))

                    bboxes.append(torch.as_tensor(get_bboxes_from_mask(nonzero_labels[idx], offset=0)))

                    nonzero_category.append(self.target_list[region_id])

                item['gt'] = nonzero_labels
                item['ori_gt'] = torch.stack(nonzero_ori_labels, dim=0)

                item['gt_target'] = nonzero_category
                item['gt_point_coords'] = torch.stack(point_coords)
                item['gt_point_labels'] = torch.stack(point_labels)
                item['gt_bboxes'] = torch.stack(bboxes)

                item['image_root'] = [image_path]
                item['dataset_name'] = [self.dataset_name]
            else:
                pseudo_path = os.path.join(self.data_dir, item_dict['imask'])

                try:
                    pseudo_array = np.load(pseudo_path).astype(np.float32)
                except:
                    print(f'{pseudo_path} not load')
                    raise ValueError(f'{pseudo_path} not load')

                item_ori = {'image': image_array, 'label': label_array, 'pseudo': pseudo_array}
                # item_ori = {key: torch.tensor(value).cuda() for key, value in item_ori.items()}
                item = self.transform(item_ori)
                item['pseudo'] = self.cleanse_pseudo_label(item['pseudo'])

                pseudo_ids = torch.unique(item['pseudo'])
                pseudo_ids = pseudo_ids[pseudo_ids != -1]

                if len(pseudo_ids) == 0:
                    return self.__getitem__(np.random.randint(self.__len__()))

                _, H, W = item['image'].shape
                select_pseudo = torch.zeros(self.mask_num, 1, H, W)

                (
                    select_pseudo, 
                    point_coords_pseudo, 
                    point_labels_pseudo, 
                    bboxes_pseudo
                    ) = self.preprocess_pseudo(item['pseudo'], pseudo_ids, select_pseudo)
                

                label_ids = torch.sum(item['label'], dim=(1,2))
                label_ids = torch.nonzero(label_ids != 0, as_tuple=True)[0].tolist()

                if len(label_ids) == 0:
                    return self.__getitem__(np.random.randint(self.__len__()))


                select_labels = torch.zeros(self.mask_num, 1, H, W)
                (
                    select_labels, 
                    point_coords, 
                    point_labels, 
                    bboxes, 
                    nonzero_category
                    ) = self. preprocess_label(item['label'], label_ids, select_labels)


                item['gt'] = select_labels
                item['pseudo'] = select_pseudo

                item['gt_point_coords'] = point_coords
                item['gt_point_labels'] = point_labels
                item['gt_bboxes'] = bboxes
                item['gt_target'] = nonzero_category
                item['pseudo_point_coords'] = point_coords_pseudo
                item['pseudo_point_labels'] = point_labels_pseudo
                item['pseudo_bboxes'] = bboxes_pseudo

            if type(item) == list:
                assert len(item) == 1
                item = item[0]
            
            assert type(item) != list
    
            post_item = self.std_keys(item)
            
        return post_item
    
    def get_preprocess_shape(self, oldh: int, oldw: int, long_side_length: int):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


    def preprocess_pseudo(self, pseudo_label, pseudo_ids, select_pseudo):
        point_coords, point_labels, bboxes = [], [], []
  
        pseudo_region_ids = random.sample(list(pseudo_ids), k=self.mask_num) if len(pseudo_ids) >= self.mask_num else random.choices(list(pseudo_ids), k=self.mask_num)
        for idx, region_id in enumerate(pseudo_region_ids):
            select_pseudo[idx][pseudo_label==region_id.item()] = 1

            point_and_labels = get_points_from_mask(select_pseudo[idx], top_num=0.5)
            point_coords.append(torch.as_tensor(point_and_labels[0]))
            point_labels.append(torch.as_tensor(point_and_labels[1]))

            bboxes.append(torch.as_tensor(get_bboxes_from_mask(select_pseudo[idx], offset=5)))

        point_coords = torch.stack(point_coords)
        point_labels = torch.stack(point_labels)
        bboxes = torch.stack(bboxes)

        return select_pseudo, point_coords, point_labels, bboxes

    def preprocess_label(self, gt_label, label_ids, select_labels):
        point_coords, point_labels, bboxes, categories = [], [], [], []
        label_region_ids = random.sample(list(label_ids), k=self.mask_num) if len(label_ids) >= self.mask_num else random.choices(list(label_ids), k=self.mask_num)
        
        for idx, region_id in enumerate(label_region_ids):
            select_labels[idx][0] = gt_label[region_id]

            point_and_labels = get_points_from_mask(select_labels[idx], top_num=0.5)
            point_coords.append(torch.as_tensor(point_and_labels[0]))
            point_labels.append(torch.as_tensor(point_and_labels[1]))

            bboxes.append(torch.as_tensor(get_bboxes_from_mask(select_labels[idx], offset=5)))
            categories.append(self.target_list[region_id])

        point_coords = torch.stack(point_coords)
        point_labels = torch.stack(point_labels)
        bboxes = torch.stack(bboxes)

        return select_labels, point_coords, point_labels, bboxes, categories


    def std_keys(self, post_item):
        keys_to_remain = ['image', 'gt', 'ori_gt', 'image_root',  'dataset_name',
                          'gt_point_coords', 'gt_point_labels', 'gt_bboxes', 'gt_target', 
                          'pseudo', 'pseudo_point_coords','pseudo_point_labels', 'pseudo_bboxes']
        keys_to_remove = post_item.keys() - keys_to_remain
        for key in keys_to_remove:
            del post_item[key]
        return post_item


    def cleanse_pseudo_label(self, pseudo_seg):
        total_voxels = pseudo_seg.numel()
        threshold = total_voxels * 0.0005
        unique_values = torch.unique(pseudo_seg)

        for value in unique_values:
            voxel_count = (pseudo_seg == value).sum()
            if voxel_count < threshold:
                pseudo_seg[pseudo_seg == value] = -1

        for label in torch.unique(pseudo_seg):
            if label == -1:
                continue

            binary_mask = pseudo_seg == label
            open = binary_opening(binary_mask.squeeze())
            close = binary_closing(open)
            processed_mask = torch.tensor(close)

            labeled_mask, num_labels = label_structure(processed_mask)
            label_sizes = sum_structure(processed_mask, labeled_mask, range(num_labels + 1))
            small_labels = np.where(label_sizes < threshold)[0]
            for label_del in small_labels:
                processed_mask[labeled_mask == label_del] = False

            pseudo_seg[binary_mask] = -1
            pseudo_seg[processed_mask.unsqueeze(0)] = label

        return pseudo_seg


def test_collate_fn(batch):
    # assert len(batch) == 1, 'Please set batch size to 1 when testing mode'
    gt_prompt = {'point_coords': [], 'point_labels': [], 'bboxes': []}
    gt_prompt['point_coords'] = batch[0]['gt_point_coords']
    gt_prompt['point_labels'] = batch[0]['gt_point_labels']
    gt_prompt['bboxes'] = batch[0]['gt_bboxes']
    image_root = batch[0]['image_root']
    target_list = batch[0]['gt_target']
    dataset_name = batch[0]['dataset_name']
    return {
        'image': batch[0]['image'].unsqueeze(0),
        'label': batch[0]['gt'],
        'ori_label': batch[0]['ori_gt'],
        'gt_prompt': gt_prompt,
        'target_list': target_list,
        'image_root': image_root,
        'dataset_name': dataset_name
    }



def train_collate_fn(batch):
    # batch_input: a dict {dict_keys(['image', 'label', 'pseudo', 'target_list', 'gt_prompt', 'pseudo_prompt'])}









    
    images, labels, pseudos, target_list = [], [], [], []
    gt_prompt = {'point_coords': [], 'point_labels': [], 'bboxes': []}
    pseudo_prompt = {'point_coords': [], 'point_labels': [], 'bboxes': []}
    
    for sample in batch:
        images.append(sample['image'])
        labels.append(sample['gt'])
        gt_prompt['point_coords'].append(sample['gt_point_coords'])
        gt_prompt['point_labels'].append(sample['gt_point_labels'])
        gt_prompt['bboxes'].append(sample['gt_bboxes'])
        target_list += sample['gt_target']

        pseudos.append(sample['pseudo'])
        pseudo_prompt['point_coords'].append(sample['pseudo_point_coords'])
        pseudo_prompt['point_labels'].append(sample['pseudo_point_labels'])
        pseudo_prompt['bboxes'].append(sample['pseudo_bboxes'])

    images = torch.stack(images, dim=0)
    labels = torch.cat(labels, dim=0)
    pseudos = torch.cat(pseudos, dim=0)
    
    gt_prompt = {key: torch.cat(value, dim=0) if len(value) !=0 else None for key, value in gt_prompt.items()}
    pseudo_prompt = {key: torch.cat(value, dim=0) if len(value) !=0 else None for key, value in pseudo_prompt.items()}

    return {
        'image': images,
        'label': labels,
        'pseudo': pseudos,
        'target_list': target_list,
        'gt_prompt': gt_prompt,
        'pseudo_prompt': pseudo_prompt,
    }
    
    
def train_collate_fn_sammed(batch):
    images, labels = [], []
    prompt = {'point_coords': [], 'point_labels': [], 'bboxes': []}
    # pseudo_prompt = {'point_coords': [], 'point_labels': [], 'bboxes': []}
    
    for sample in batch:
        images.append(sample['image'])
        
        assert sample['gt'].shape == sample['pseudo'].shape, 'GT and pseudo shape should be the same'
        
        labels.append(sample['gt'])
        # labels.append(sample['pseudo'])
        
        prompt['point_coords'].append(sample['gt_point_coords'])
        prompt['point_labels'].append(sample['gt_point_labels'])
        prompt['bboxes'].append(sample['gt_bboxes'])
        
        # prompt['point_coords'].append(sample['pseudo_point_coords'])
        # prompt['point_labels'].append(sample['pseudo_point_labels'])
        # prompt['bboxes'].append(sample['pseudo_bboxes'])
        
    images = torch.stack(images, dim=0)
    labels = torch.cat(labels, dim=0)
    
    prompt = {key: torch.cat(value, dim=0) if len(value) !=0 else None for key, value in prompt.items()}

    return {
        'image': images,
        'label': labels,
        'prompt': prompt,
    }
        

def get_loader(args, all_training_sets=None):

    all_data_set = []
    for sub_dataset in all_training_sets:
        
        if sub_dataset == 'sa000000':
            target_size = (args.image_size, args.image_size)
            transform = transforms.Compose(
                            [
                                Resize(keys=["image", "label"], target_size=target_size), 
                                PermuteTransform(keys=["image"], dims=(2,0,1)),
                                transforms.ToTensord(keys=["image", "label"]),
                                Normalization(keys=["image"]),
                            ]
                        )
            collate_fn = test_collate_fn
            
            dataset = UniversalDataset(
                dataset_name=sub_dataset,
                args=args, 
                datalist=[],
                classes_list=[],
                transform = transform
                )
        else:
            sub_dataset_root = os.path.join(args.data_dir, sub_dataset)
            dataset_json = os.path.join(sub_dataset_root, 'dataset.json')
            dataset_dict = json.load(open(dataset_json, 'r'))
            target_size = (args.image_size, args.image_size)

            if args.test_mode:
                datalist = dataset_dict['test']
                collate_fn = test_collate_fn
                transform = transforms.Compose(
                                [
                                    Resize(keys=["image", "label"], target_size=target_size), 
                                    PermuteTransform(keys=["image"], dims=(2,0,1)),
                                    transforms.ToTensord(keys=["image", "label"]),
                                    Normalization(keys=["image"]),
                                ]
                            )

            else:
                datalist = dataset_dict['training']
                # shuffle datalist
                random.shuffle(datalist)
                datalist_len = len(datalist)
                selected_len = math.ceil(datalist_len * args.dataset_scale)
                datalist = datalist[:selected_len]
                # datalist = random.sample(datalist, k=math.ceil(len(datalist) * args.dataset_scale))
                collate_fn = train_collate_fn_sammed
                transform = transforms.Compose(
                        [
                            Resize(keys=["image", "label", "pseudo"], target_size=target_size),  #
                            PermuteTransform(keys=["image"], dims=(2,0,1)),
                            transforms.ToTensord(keys=["image", "label", "pseudo"]),
                            Normalization(keys=["image"]),
                            transforms.RandScaleIntensityd(keys="image", factors=0.2, prob=0.2),
                            transforms.RandShiftIntensityd(keys="image", offsets=0.2, prob=0.2),
                        ]
                    )
        
            classes_list = list(dataset_dict['labels'].values())

            dataset = UniversalDataset(
                dataset_name=sub_dataset,
                args=args, 
                datalist=datalist, 
                classes_list=classes_list,
                transform = transform
                )
        
        all_data_set.append(dataset)

    # concat all datasets to one
    print(f"Dataset and their len: {[(dataset.dataset_name, len(dataset)) for dataset in all_data_set]}")
    dataset = torch.utils.data.ConcatDataset(all_data_set)
    print(f"Total length of dataset: {len(dataset)}")
    
    # sampler = DistributedSampler(dataset) if args.dist else None

    data_loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        # shuffle=(sampler is None),
        shuffle=False if args.test_mode else True,
        num_workers=args.num_workers,
        sampler=None,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
    )

    return data_loader

def get_subset_loader(dataset_name: str, data_dir: str, n_samples: int | None):
    full_ds = get_loader(args, all_training_sets=[dataset_name], return_dataset=True)
    if n_samples is None:
        return full_ds
    n = len(full_ds)
    k = min(n_samples, n)
    idx = np.random.choice(n, k, replace=False).tolist()
    return torch.utils.data.Subset(full_ds, idx)


import copy
from typing import List, Tuple

def _build_single_dataset(args, dataset_name: str, *, force_test_mode: bool | None = None,
                          force_dataset_scale: float | None = None):

    _args = copy.deepcopy(args)
    if force_test_mode is not None:
        _args.test_mode = force_test_mode
    if force_dataset_scale is not None:
        _args.dataset_scale = force_dataset_scale

    target_size = (_args.image_size, _args.image_size)

    if dataset_name == 'sa000000':
        transform = transforms.Compose([
            Resize(keys=["image", "label"], target_size=target_size),
            PermuteTransform(keys=["image"], dims=(2,0,1)),
            transforms.ToTensord(keys=["image", "label"]),
            Normalization(keys=["image"]),
        ])
        collate_fn = test_collate_fn
        dataset = UniversalDataset(
            dataset_name=dataset_name,
            args=_args,
            datalist=[],
            classes_list=[],
            transform=transform
        )
        return dataset, collate_fn


    sub_dataset_root = os.path.join(_args.data_dir, dataset_name)
    dataset_json = os.path.join(sub_dataset_root, 'dataset.json')
    dataset_dict = json.load(open(dataset_json, 'r'))
    classes_list = list(dataset_dict['labels'].values())

    if _args.test_mode:
        datalist = dataset_dict['test']
        collate_fn = test_collate_fn
        transform = transforms.Compose([
            Resize(keys=["image", "label"], target_size=target_size),
            PermuteTransform(keys=["image"], dims=(2,0,1)),
            transforms.ToTensord(keys=["image", "label"]),
            Normalization(keys=["image"]),
        ])
    else:
        datalist = dataset_dict['training']
        random.shuffle(datalist)

        if force_dataset_scale is not None:
            selected_len = math.ceil(len(datalist) * force_dataset_scale)
            datalist = datalist[:selected_len]
        else:
            selected_len = math.ceil(len(datalist) * _args.dataset_scale)
            datalist = datalist[:selected_len]

        collate_fn = train_collate_fn_sammed
        transform = transforms.Compose([
            Resize(keys=["image", "label", "pseudo"], target_size=target_size),
            PermuteTransform(keys=["image"], dims=(2,0,1)),
            transforms.ToTensord(keys=["image", "label", "pseudo"]),
            Normalization(keys=["image"]),
            transforms.RandScaleIntensityd(keys="image", factors=0.2, prob=0.2),
            transforms.RandShiftIntensityd(keys="image", offsets=0.2, prob=0.2),
        ])

    dataset = UniversalDataset(
        dataset_name=dataset_name,
        args=_args,
        datalist=datalist,
        classes_list=classes_list,
        transform=transform
    )
    return dataset, collate_fn


def build_membank_dataset(args, dataset_name: str, k_samples: int, seed: int = 0):

    base_ds, collate_fn = _build_single_dataset(
        args, dataset_name,
        force_test_mode=False,
        force_dataset_scale=1.0
    )
    n = len(base_ds)
    if n == 0:
        raise RuntimeError(f"[membank] Dataset {dataset_name} has length 0.")

    g = torch.Generator()
    g.manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:min(k_samples, n)].tolist()

    subset = torch.utils.data.Subset(base_ds, idx)

    return subset, train_collate_fn_sammed


def build_train_dataset_with_er(args, cur_train_sets: List[str],
                                membank_specs: List[Tuple[str, int]],
                                seed: int = 0):

    cur_datasets = []
    for name in cur_train_sets:
        ds, collate_fn = _build_single_dataset(args, name, force_test_mode=False)

        assert collate_fn is train_collate_fn_sammed, \
            "Current train collate_fn should be train_collate_fn_sammed"
        cur_datasets.append(ds)


    membank_datasets = []
    for name, k in membank_specs:
        sub, _ = build_membank_dataset(args, name, k, seed=seed)
        membank_datasets.append(sub)


    all_ds = cur_datasets + membank_datasets
    if len(all_ds) == 0:
        raise RuntimeError("No datasets found for training with ER.")
    concat_ds = torch.utils.data.ConcatDataset(all_ds)
    return concat_ds, train_collate_fn_sammed


def build_train_loader_with_er(args, cur_train_sets: List[str],
                               membank_specs: List[Tuple[str, int]],
                               seed: int = 0):
    dataset, collate_fn = build_train_dataset_with_er(args, cur_train_sets, membank_specs, seed=seed)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
    )
    return loader
# ============================================================================


if __name__ == "__main__":
    import argparse
    multiprocessing.set_start_method('spawn')
    # dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
    def set_parse():
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_dir", type=str, default='./Med_datasets')
        parser.add_argument('--image_size', type=int, default=1024)
        parser.add_argument('--test_mode', type=bool, default=False)
        parser.add_argument('--batch_size', type=int, default=1)
        # parser.add_argument('--dist', dest='dist', type=bool, default=True,help='distributed training or not')
        parser.add_argument('--num_workers', type=int, default=10)
        parser.add_argument('--mask_num', type=int, default=5)
        parser.add_argument('--dataset_scale', type=float, default=1)
        args = parser.parse_args()
        return args
    args = set_parse()
    args.test_mode = True
    # train_loader = get_loader(args, all_training_sets=['WORD', 'AMOS2022_CT', 'BraTS2020', 'MMWHS_MR', 'TDSC-ABUS2023', 'kvasir_seg_aliyun'])
    # print("Length of dataloader: ", len(train_loader))
    # for idx, batch in enumerate(train_loader):
    #     image, label = batch["image"], batch["label"]
    #     # pseudo = batch['pseudo']
    #test_loader = get_loader(args, all_training_sets=['ACDC', 'innerdata/WORD', 'innerdata/AMOS2022_CT', 'innerdata/BraTS2020', 
     #                                                 'innerdata/MMWHS_MR', 'innerdata/TDSC-ABUS2023', 'innerdata/kvasir_seg_aliyun',
      #                                                'zeroshotdata/ACDC', 'zeroshotdata/BTCV', 'zeroshotdata/COVID19CTscans', 'zeroshotdata/gamma'])
    test_loader = get_loader(args, all_training_sets=['ACDC', 'Ployp', 'BSData', 'MVTec_1', 'MVTec_2', 'MVTec_3', 'RSDDs_1', 'RSDDs_2', 'CAMO', 'innerdata/WORD', 'innerdata/AMOS2022_CT', 'innerdata/BraTS2020', 
                                                      'innerdata/MMWHS_MR', 'innerdata/TDSC-ABUS2023', 'innerdata/kvasir_seg_aliyun',
                                                      'zeroshotdata/ACDC', 'zeroshotdata/BTCV', 'zeroshotdata/COVID19CTscans', 'zeroshotdata/gamma'])
    print("Length of dataloader: ", len(test_loader))
    for idx, batch in enumerate(test_loader):
        image, label = batch["image"], batch["label"]
        # pseudo = batch['pseudo']
        print(image.shape, label.shape)
        break
