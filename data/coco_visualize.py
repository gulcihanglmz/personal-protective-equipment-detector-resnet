from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import cv2
import random
import os

DATA_DIR      = r"C:\SH17-dataset\root"
IMAGES_DIR    = os.path.join(DATA_DIR, "images")
ANNOTATIONS   = os.path.join(DATA_DIR, "annotations", "train.json")

coco = COCO(ANNOTATIONS)

img_ids = coco.getImgIds()
for _ in range(30):
    img_id   = random.choice(img_ids)
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(IMAGES_DIR, img_info["file_name"])
    img      = cv2.imread(img_path)
    img      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns    = coco.loadAnns(ann_ids)

    plt.figure(figsize=(12,8))
    plt.imshow(img)
    plt.axis("off")

    for ann in anns:
        x,y,w,h = ann["bbox"]
        cat     = coco.loadCats(ann["category_id"])[0]["name"]
        # kutu
        plt.gca().add_patch(
            plt.Rectangle(
                (x, y), w, h,
                fill=False, edgecolor="lime", linewidth=2
            )
        )
        plt.text(
            x, y-5, cat,
            color="black", fontsize=12,
            backgroundcolor="yellow"
        )

        if "segmentation" in ann and isinstance(ann["segmentation"], list):
            for seg in ann["segmentation"]:
                poly = plt.Polygon(
                    [seg[i:i+2] for i in range(0, len(seg), 2)],
                    facecolor="none", edgecolor="red", linewidth=1
                )
                plt.gca().add_patch(poly)

    out_path = os.path.join(DATA_DIR, "visualizations", f"{img_id}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")

    plt.show()
