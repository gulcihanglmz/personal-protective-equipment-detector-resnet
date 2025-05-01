import torch
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

def evaluate(model, data_loader, ann_file, device):
    model.eval()
    coco_gt = COCO(ann_file)
    results = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            # PyCOCOtools formatına dönüştür
            for tgt, out in zip(targets, outputs):
                img_id = tgt["image_id"].item()
                for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
                    bbox = box.tolist()
                    results.append({
                        "image_id": img_id,
                        "category_id": label.item(),
                        "bbox": [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]],
                        "score": score.item()
                    })

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats  # mAP değerleri
