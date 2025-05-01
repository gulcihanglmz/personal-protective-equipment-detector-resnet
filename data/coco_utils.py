import os
import random
import torch
from PIL import Image
from pycocotools.coco import COCO
from torchvision.datasets import VisionDataset
import torchvision.transforms.functional as F

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        # PIL image → CxHxW float tensor in [0,1]
        if isinstance(image, Image.Image):
            arr = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
            arr = arr.view(image.size[1], image.size[0], len(image.getbands()))
            tensor = arr.permute(2, 0, 1).float().div(255)
        else:
            tensor = torch.as_tensor(image, dtype=torch.float32)
            if tensor.ndim == 3:
                tensor = tensor.permute(2, 0, 1).contiguous()
        return tensor, target

class RandomHorizontalFlip:
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            w = image.shape[2]
            boxes = target["boxes"]
            new_boxes = boxes.clone()
            new_boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            target["boxes"] = new_boxes
        return image, target

class Resize:
    def __init__(self, size):
        self.size = size  # (H, W)

    def __call__(self, image, target):
        image = F.resize(image, self.size)
        w_old, h_old = image.size
        w_new, h_new = self.size[1], self.size[0]
        scale_x, scale_y = w_new / w_old, h_new / h_old
        boxes = target["boxes"]
        boxes[:, [0,2]] *= scale_x
        boxes[:, [1,3]] *= scale_y
        target["boxes"] = boxes
        return image, target

def get_transform(train):
    transforms = [Resize((800,800)), ToTensor()]
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)

class COCODetectionDataset(VisionDataset):
    def __init__(self, root, ann_file, transforms=None):
        super().__init__(root, transforms=transforms)
        self.root = root
        ann_path = os.path.join(root, ann_file)
        self.coco = COCO(ann_path)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        img_info = coco.loadImgs(img_id)[0]
        path = img_info["file_name"]
        image = Image.open(os.path.join(self.root, "images", path)).convert("RGB")

        boxes = [obj["bbox"] for obj in anns]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        boxes[:, 2:] += boxes[:, :2]
        labels = torch.as_tensor([obj["category_id"] for obj in anns], dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def __len__(self):
        return len(self.ids)

def make_coco_dataset(root, ann_file, train=True):
    return COCODetectionDataset(
        root=root,
        ann_file=os.path.join("annotations", ann_file),
        transforms=get_transform(train),
    )
