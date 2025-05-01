# utils/visualizer.py
import matplotlib.pyplot as plt

def draw_predictions(img, outputs, coco, score_thr=0.5):
    plt.imshow(img)
    plt.axis("off")
    for box, label, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
        if score < score_thr:
            continue
        x,y,x2,y2 = box
        cat = coco.loadCats(label.item())[0]["name"]
        plt.gca().add_patch(
            plt.Rectangle((x,y), x2-x, y2-y, fill=False, edgecolor="cyan", linewidth=2)
        )
        plt.text(x, y-5, f"{cat} {score:.2f}", color="white",
                 bbox=dict(facecolor="black", alpha=0.6, pad=2))
    plt.show()
