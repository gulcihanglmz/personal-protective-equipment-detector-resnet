# visualizer.py
import matplotlib.pyplot as plt
from data.coco_utils import make_coco_dataset
from pycocotools.coco import COCO


def draw_ground_truth(img, target):
    """
    Gerçek (ground-truth) kutuları kırmızı renkte çizer.
    """
    plt.imshow(img.permute(1, 2, 0).cpu())
    plt.axis("off")
    for box in target["boxes"]:
        x, y, x2, y2 = box
        plt.gca().add_patch(
            plt.Rectangle((x, y), x2 - x, y2 - y,
                          fill=False, edgecolor="red", linewidth=2)
        )
    plt.show()


def draw_predictions(img, outputs, coco, score_thr=0.5):
    """
    Model çıktısını (predictions) cyan renkte çizer.
    - img: Tensor[C,H,W]
    - outputs: model(img) çıktısı dict (boxes, labels, scores)
    - coco: pycocotools COCO nesnesi, label → kategori adı çözümlemek için
    """
    plt.imshow(img.permute(1, 2, 0).cpu())
    plt.axis("off")
    for box, label, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
        if score < score_thr:
            continue
        x, y, x2, y2 = box
        cat = coco.loadCats(label.item())[0]["name"]
        plt.gca().add_patch(
            plt.Rectangle((x, y), x2 - x, y2 - y,
                          fill=False, edgecolor="cyan", linewidth=2)
        )
        plt.text(x, y - 5, f"{cat} {score:.2f}", color="white",
                 bbox=dict(facecolor="black", alpha=0.6, pad=2))
    plt.show()


if __name__ == "__main__":
    # Örnek kullanım:
    cfg = __import__("yaml").safe_load(open("configs/default.yaml"))
    # Dataset ve COCO objesini oluştur
    ds = make_coco_dataset(cfg["DATA_DIR"], cfg["TRAIN_JSON"], train=True)
    coco = COCO(f"{cfg['DATA_DIR']}/annotations/{cfg['TRAIN_JSON']}")

    # 0. indeksteki görüntü ve anotasyonu al
    img, target = ds[0]
    draw_ground_truth(img, target)

    # Model çıktısını simüle etmek için:
    # (Gerçekte model = get_model(...).eval(); outputs = model([img.to(device)])[0])
    # Burada dummy olarak ground-truth’u baz alıyoruz
    outputs = {
        "boxes": target["boxes"],
        "labels": target["labels"],
        "scores": torch.ones(len(target["boxes"]))
    }
    draw_predictions(img, outputs, coco, score_thr=0.5)
