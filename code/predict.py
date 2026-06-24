import argparse
import csv
import colorsys
from pathlib import Path
 
import cv2
from tqdm import tqdm
from ultralytics import YOLO
 
 
# ─────────────────────────────────────────────────────────────
# 클래스 ID → 고유 색상 생성 (HSV 균등 분할 → BGR 변환)
# n_colors 만큼 색상환을 균등 분할하므로 몇 개든 겹치지 않음
# ─────────────────────────────────────────────────────────────
def generate_colors(n_colors: int) -> list[tuple[int, int, int]]:
    colors = []
    for i in range(n_colors):
        hue = i / n_colors                  # 0.0 ~ 1.0 균등 분할
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))  # BGR
    return colors
 
 
def get_color(cls_id: int, color_map: dict, n_base: int) -> tuple[int, int, int]:
    """
    color_map 에 없는 cls_id (모델 학습 외 클래스) 는
    n_base 이후 구간에서 색상을 동적으로 생성해 할당한다.
    """
    if cls_id not in color_map:
        # 모델이 모르는 클래스 → 회색 계열로 표시
        gray = 80 + (cls_id * 37) % 120   # 80~199 사이 회색 변주
        color_map[cls_id] = (gray, gray, gray)
    return color_map[cls_id]
 
 
def draw_boxes(image, boxes, class_names: dict, color_map: dict, n_base: int):
    """이미지에 bbox, 클래스명, confidence를 그린다."""
    for box in boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
 
        color = get_color(cls_id, color_map, n_base)
 
        # 모델이 아는 클래스면 이름 표시, 모르면 "unknown_{id}" 표시
        cls_name = class_names.get(cls_id, f"unknown_{cls_id}")
        label    = f"{cls_name} {conf:.2f}"
 
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
 
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    return image
 
 
def run_predict(model_path: str, source: str, output_dir: str, conf: float):
    model       = YOLO(model_path)
    class_names = model.names          # {0: 'class_a', 1: 'class_b', ...}
    n_classes   = len(class_names)     # 학습된 클래스 수 (예: 51)
 
    # 학습된 클래스 수 기준으로 색상 생성 → cls_id별 고유 색상 매핑
    palette   = generate_colors(n_classes)
    color_map = {cls_id: palette[cls_id] for cls_id in range(n_classes)}
    # 모델 밖 클래스는 get_color() 내부에서 회색으로 동적 추가됨
 
    src_dir = Path(source)
    exts    = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images  = sorted([p for p in src_dir.rglob("*") if p.suffix.lower() in exts])
 
    if not images:
        print(f"❌ 이미지를 찾을 수 없습니다: {src_dir}")
        return
 
    # 출력 폴더
    out_root  = Path(output_dir)
    img_dir   = out_root / "images"
    label_dir = out_root / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
 
    no_det_list    = []
    error_list     = []
    total_pred     = 0
    unknown_ids    = set()   # 모델이 모르는 cls_id 수집
 
    print(f"\n🔍 총 {len(images)}장 추론 시작")
    print(f"   모델 클래스 수 : {n_classes}개")
    print(f"   conf threshold : {conf}\n")
 
    for img_path in tqdm(images, unit="img", dynamic_ncols=True):
        try:
            results = model.predict(
                source=str(img_path),
                conf=conf,
                verbose=False,
            )
            result  = results[0]
            boxes   = result.boxes
            n_boxes = len(boxes) if boxes is not None else 0
            total_pred += n_boxes
 
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"cv2.imread 실패: {img_path}")
 
            if n_boxes > 0:
                # 모델 밖 클래스 ID 수집
                for box in boxes:
                    cid = int(box.cls[0])
                    if cid not in class_names:
                        unknown_ids.add(cid)
 
                img = draw_boxes(img, boxes, class_names, color_map, n_classes)
 
            # 이미지 저장
            cv2.imwrite(str(img_dir / img_path.name), img)
 
            # YOLO txt 저장
            txt_path = label_dir / (img_path.stem + ".txt")
            if n_boxes > 0:
                h, w = img.shape[:2]
                with txt_path.open("w") as f:
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cx = ((x1 + x2) / 2) / w
                        cy = ((y1 + y2) / 2) / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            else:
                txt_path.touch()
                no_det_list.append(img_path.name)
 
        except Exception as e:
            error_list.append({"image": img_path.name, "error": str(e)})
            tqdm.write(f"  ⚠️  SKIP [{img_path.name}]: {e}")
            continue
 
    # No Detection 목록 저장
    no_det_log = out_root / "no_detection.txt"
    with no_det_log.open("w", encoding="utf-8") as f:
        f.write(f"# No Detection 이미지 목록 ({len(no_det_list)}장)\n")
        for name in no_det_list:
            f.write(name + "\n")
 
    # 에러 목록 저장
    if error_list:
        err_log = out_root / "errors.csv"
        with err_log.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "error"])
            writer.writeheader()
            writer.writerows(error_list)
 
    # 색상 범례 저장 (어떤 cls_id가 무슨 색인지)
    legend_path = out_root / "color_legend.txt"
    with legend_path.open("w", encoding="utf-8") as f:
        f.write("# cls_id | 클래스명 | BGR 색상\n")
        for cid, color in sorted(color_map.items()):
            name = class_names.get(cid, f"[unknown_{cid}]")
            f.write(f"{cid:>4}  {name:<30}  BGR={color}\n")
 
    # 최종 요약
    processed = len(images) - len(error_list)
    print("\n" + "=" * 58)
    print("  추론 완료 요약")
    print("=" * 58)
    print(f"  전체 이미지          : {len(images):>6}장")
    print(f"  처리 완료            : {processed:>6}장")
    print(f"  No Detection         : {len(no_det_list):>6}장")
    print(f"  에러(스킵)           : {len(error_list):>6}장")
    print(f"  총 예측 box 수       : {total_pred:>6}개")
    if unknown_ids:
        print(f"  ⚠️  모델 밖 cls_id   : {sorted(unknown_ids)}  ← 회색 표시")
    print("=" * 58)
    print(f"  📁 이미지    → {img_dir.resolve()}")
    print(f"  📁 라벨      → {label_dir.resolve()}")
    print(f"  📄 No-Det    → {no_det_log.resolve()}")
    print(f"  📄 색상 범례 → {legend_path.resolve()}")
    if error_list:
        print(f"  📄 에러      → {(out_root / 'errors.csv').resolve()}")
    print()
 
    return label_dir
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 예측 실행")
    parser.add_argument("--model",  default=r"C:\Users\Administrator\pill\base\exp_results\SGD_exp01_base\weights\best.pt", help="가중치 경로 (.pt)")
    parser.add_argument("--source", default=r"C:\Users\Administrator\pill\base\test\images", help="테스트 이미지 폴더")
    parser.add_argument("--output", default=r"C:\Users\Administrator\pill\base\predictions", help="결과 저장 루트 폴더")
    parser.add_argument("--conf",   type=float, default=0.25, help="confidence threshold")
    args = parser.parse_args()
 
    run_predict(args.model, args.source, args.output, args.conf)