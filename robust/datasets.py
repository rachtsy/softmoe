# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json


from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader


from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

# dont think this is in use anymore
PATH_TO_IMAGENET_VAL = '/path/to/data/imagenet/val'

# dont think this is in use anymore
def create_symlinks_to_imagenet(imagenet_folder, folder_to_scan):
    if not os.path.exists(imagenet_folder):
        os.makedirs(imagenet_folder)
        folders_of_interest = os.listdir(folder_to_scan)
        path_prefix = PATH_TO_IMAGENET_VAL
        for folder in folders_of_interest:
            os.symlink(path_prefix + folder, imagenet_folder+folder, target_is_directory=True)




class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)


        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)


        path_json_for_targeter = os.path.join(root, f"train{year}.json")


        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)


        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)


        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])


            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))


    # __getitem__ and __len__ inherited from ImageFolder



# called from main twice, once for training, once for val
def build_dataset(is_train, args):
    transform = build_transform(is_train, args)


    if args.data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    if args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform)
        nb_classes = 10
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        class_names = dataset.classes
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes



    return dataset, nb_classes

try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR


    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp


# def build_transform(is_train, config):
#     resize_im = config.DATA.IMG_SIZE > 32
#     if is_train:
#         # this should always dispatch to transforms_imagenet_train
#         transform = create_transform(
#             input_size=config.DATA.IMG_SIZE,
#             is_training=True,
#             color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
#             auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
#             re_prob=config.AUG.REPROB,
#             re_mode=config.AUG.REMODE,
#             re_count=config.AUG.RECOUNT,
#             interpolation=config.DATA.INTERPOLATION,
#         )
#         if not resize_im:
#             # replace RandomResizedCropAndInterpolation with
#             # RandomCrop
#             transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
#         return transform

#     t = []
#     if resize_im:
#         if config.TEST.CROP:
#             size = int((256 / 224) * config.DATA.IMG_SIZE)
#             t.append(
#                 transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
#                 # to maintain same ratio w.r.t. 224 images
#             )
#             t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
#         else:
#             t.append(
#                 transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
#                                   interpolation=_pil_interp(config.DATA.INTERPOLATION))
#             )

#     t.append(transforms.ToTensor())
#     t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
#     return transforms.Compose(t)


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform


    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))


    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)



