import argparse
import csv
from collections import defaultdict
from pathlib import Path
 
# ── 학습한 51개 클래스 ID ──────────────────────────────────
TRAIN_CLASS_IDS = {
    1, 4, 8, 10, 12, 13, 14, 15, 16, 23, 25, 31, 33, 36, 37, 38,
    40, 41, 42, 43, 45, 46, 47, 48, 50, 51, 54, 55, 56, 57, 58, 64,
    68, 70, 72, 75, 78, 79, 85, 86, 87, 89, 90, 91, 92, 95, 96, 97,
    99, 103, 104
}
 
 
def analyze(gt_dir: str, output_csv: str):
    gt_path = Path(gt_dir)
    txt_files = sorted(gt_path.glob("*.txt"))
 
    if not txt_files:
        print(f"❌ txt 파일을 찾을 수 없습니다: {gt_path}")
        return
 
    total_boxes        = 0   # GT 전체 box
    total_train_boxes  = 0   # 51개 클래스 box
    total_other_boxes  = 0   # 나머지 클래스 box
 
    has_train_cls  = 0   # 51개 클래스 box가 1개 이상인 이미지
    no_train_cls   = 0   # 51개 클래스 box가 0개인 이미지
    empty_files    = 0   # 아예 빈 파일 (box 없음)
 
    class_counter  = defaultdict(int)   # cls_id → box 수
    per_image_rows = []                 # 이미지별 상세
 
    for txt_file in txt_files:
        lines = [l.strip() for l in txt_file.read_text().splitlines() if l.strip()]
        n_all   = len(lines)
        n_train = 0
 
        for line in lines:
            try:
                cls_id = int(line.split()[0])
                class_counter[cls_id] += 1
                if cls_id in TRAIN_CLASS_IDS:
                    n_train += 1
            except (ValueError, IndexError):
                continue
 
        n_other = n_all - n_train
        total_boxes       += n_all
        total_train_boxes += n_train
        total_other_boxes += n_other
 
        if n_all == 0:
            empty_files += 1
        if n_train > 0:
            has_train_cls += 1
        else:
            no_train_cls += 1
 
        per_image_rows.append({
            "image":        txt_file.stem,
            "total_boxes":  n_all,
            "train_boxes":  n_train,
            "other_boxes":  n_other,
        })
 
    # ── 콘솔 출력 ─────────────────────────────────────────────
    print("\n" + "=" * 58)
    print("  GT 라벨 분석 결과")
    print("=" * 58)
    print(f"  분석 이미지 수           : {len(txt_files):>7}장")
    print(f"  빈 파일 (box 없음)       : {empty_files:>7}장")
    print(f"  51cls box 있는 이미지    : {has_train_cls:>7}장")
    print(f"  51cls box 없는 이미지    : {no_train_cls:>7}장")
    print("-" * 58)
    print(f"  GT 전체 box 수           : {total_boxes:>7}개")
    print(f"  학습 51cls box 수        : {total_train_boxes:>7}개  ← 검증 기준")
    print(f"  나머지 cls box 수        : {total_other_boxes:>7}개  (비교 대상 X)")
    print("=" * 58)
 
    # 클래스별 box 수 — 학습 클래스 / 비학습 클래스 분리 출력
    train_cls_counts = {k: v for k, v in class_counter.items() if k in TRAIN_CLASS_IDS}
    other_cls_counts = {k: v for k, v in class_counter.items() if k not in TRAIN_CLASS_IDS}
 
    print("\n  [학습 51개 클래스별 box 수]")
    print(f"  {'cls_id':>8}  {'box 수':>8}")
    print("  " + "-" * 20)
    for cls_id in sorted(train_cls_counts):
        print(f"  {cls_id:>8}  {train_cls_counts[cls_id]:>8}")
    print(f"  {'합계':>8}  {sum(train_cls_counts.values()):>8}")
 
    print("\n  [비학습 클래스별 box 수 (검증 제외)]")
    print(f"  {'cls_id':>8}  {'box 수':>8}")
    print("  " + "-" * 20)
    for cls_id in sorted(other_cls_counts):
        print(f"  {cls_id:>8}  {other_cls_counts[cls_id]:>8}")
    print(f"  {'합계':>8}  {sum(other_cls_counts.values()):>8}")
 
    # ── CSV 저장 ──────────────────────────────────────────────
    if output_csv:
        out = Path(output_csv)
 
        # 시트 1: 이미지별 상세
        with out.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=per_image_rows[0].keys())
            writer.writeheader()
            writer.writerows(per_image_rows)
 
        # 시트 2: 클래스별 집계 (같은 CSV에 이어서 작성)
        cls_csv = out.parent / (out.stem + "_class_stats.csv")
        with cls_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["cls_id", "box_count", "is_train_class"])
            for cls_id in sorted(class_counter):
                writer.writerow([
                    cls_id,
                    class_counter[cls_id],
                    "O" if cls_id in TRAIN_CLASS_IDS else "X",
                ])
 
        print(f"\n📄 이미지별 통계  → {out.resolve()}")
        print(f"📄 클래스별 통계  → {cls_csv.resolve()}")
 
    print()
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GT 라벨 box 수 분석")
    parser.add_argument("--gt",     default=r"C:\Users\Administrator\pill\base\test\labels", help="GT label 폴더 경로")
    parser.add_argument("--output", default="gt_box_stats.csv", help="결과 CSV 저장 경로")
    args = parser.parse_args()
 
    analyze(args.gt, args.output)
 