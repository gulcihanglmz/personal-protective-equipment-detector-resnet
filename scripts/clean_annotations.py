import json
from pathlib import Path

def clean_coco(json_path, out_path):
    data = json.load(open(json_path))
    imgs = {im["id"]:(im["width"], im["height"]) for im in data["images"]}
    new_anns = []
    for ann in data["annotations"]:
        x,y,w,h = ann["bbox"]
        W,H = imgs[ann["image_id"]]
        if w <= 0 or h <= 0:
            continue
        x2, y2 = x+w, y+h
        if x<0 or y<0 or x2>W or y2>H:
            x = max(0, x); y = max(0, y)
            w = min(w, W-x); h = min(h, H-y)
            if w <= 0 or h <= 0:
                continue
            ann["bbox"] = [x,y,w,h]
            ann["area"] = w*h
        new_anns.append(ann)
    data["annotations"] = new_anns
    json.dump(data, open(out_path, "w"), indent=2)

if __name__=="__main__":
    clean_coco(
      r"C:\SH17-dataset\root\annotations\result.json",
      r"C:\SH17-dataset\root\annotations\cleaned.json"
    )
    print("Annotations cleaned!")
