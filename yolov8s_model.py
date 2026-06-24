from pathlib import Path
import torch
import pandas as pd
from ultralytics import YOLO


def train_and_val(exp_cfg: dict, data_path: Path, runs_dir: Path, device: str) -> None:
    """학습 및 검증(val 데이터셋)을 진행하고 학습 Loss를 출력하는 함수"""
    exp_name = exp_cfg["name"]
    print("\n" + "=" * 60)
    print(f"Starting Training: {exp_name}")
    print("=" * 60)

    model = YOLO("yolov8s.yaml")  # scratch에서 학습
    model.train(
        data=str(data_path),
        epochs=exp_cfg["epochs"],
        imgsz=exp_cfg["imgsz"],
        batch=exp_cfg["batch"],
        device=device,
        workers=0,
        project=str(runs_dir),
        name=exp_name,
        exist_ok=True,
        pretrained=exp_cfg["pretrained"],
        patience=exp_cfg["patience"],
        optimizer=exp_cfg["optimizer"],
        lr0=exp_cfg["lr0"],
        lrf=exp_cfg["lrf"],
        cos_lr=exp_cfg["cos_lr"],
        box=exp_cfg["box"],
        cls=exp_cfg["cls"],
        dfl=exp_cfg["dfl"],
        mosaic=exp_cfg["mosaic"],
        mixup=exp_cfg["mixup"],
        copy_paste=exp_cfg["copy_paste"],
        degrees=exp_cfg["degrees"],
        translate=exp_cfg["translate"],
        scale=exp_cfg["scale"],
        fliplr=exp_cfg["fliplr"],
        flipud=exp_cfg["flipud"],
        hsv_h=0.015,
        hsv_s=exp_cfg["hsv_s"],
        hsv_v=exp_cfg["hsv_v"],
        erasing=0.4,
        close_mosaic=exp_cfg["close_mosaic"],
    )

    # Train Loss 추출
    csv_path = runs_dir / exp_name / "results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        train_box_loss = df['train/box_loss'].iloc[-1]
        train_cls_loss = df['train/cls_loss'].iloc[-1]
        print(f"\n[{exp_name}] Final Train Loss -> Box: {train_box_loss:.4f}, Cls: {train_cls_loss:.4f}")
    else:
        print("Warning: results.csv를 찾을 수 없어 Train Loss를 불러오지 못했습니다.")


def evaluate_test_set(weights_path: Path, data_path: Path, runs_dir: Path, exp_name: str, batch_size: int, device: str, target_classes: list) -> None:
    """학습된 최고 가중치(best.pt)를 불러와 지정한 특정 클래스(target_classes)만 test 데이터셋으로 평가하는 함수"""
    if not weights_path.exists():
        print(f"\n[오류] 가중치 파일을 찾을 수 없습니다: {weights_path}")
        return

    print("\n" + "=" * 60)
    print(f"Starting Test Evaluation for: {exp_name}")
    print("=" * 60)

    model = YOLO(str(weights_path))

    metrics = model.val(
        data=str(data_path),
        split="test",
        batch=batch_size,
        device=device,
        classes=target_classes,  # 이 매개변수로 특정 클래스 결과만 수집합니다.
        workers=0,
        project=str(runs_dir),
        name=f"{exp_name}_test_evaluation",
        exist_ok=True
    )

    print("\n" + "-" * 60)
    print(f"[{exp_name}] Test Dataset Evaluation Results ({len(target_classes)} Classes)")
    print("-" * 60)
    print(f"Precision (P)  : {metrics.box.mp:.4f}")
    print(f"Recall (R)     : {metrics.box.mr:.4f}")
    print(f"mAP@50         : {metrics.box.map50:.4f}")
    print(f"mAP@50-95      : {metrics.box.map:.4f}")
    print("-" * 60)


if __name__ == "__main__":
    # 1. 기본 경로 설정 (알맞게 경로를 확인하세요)
    data_path = Path(r"C:\Users\Administrator\pill\base\data.yaml")
    runs_dir  = Path(r"C:\Users\Administrator\pill\base\exp_results")
    device    = "0" if torch.cuda.is_available() else "cpu"

    print(f"Training device: {device}")
    print(f"Data config    : {data_path}")
    print(f"Results dir    : {runs_dir}")

    # 2. 평가 타겟 클래스 (요청하신 51개 클래스 번호 지정)
    target_classes = [
        1, 4, 8, 10, 12, 13, 14, 15, 16, 23, 25, 31, 33, 36, 37, 38, 40, 41, 42, 43,
        45, 46, 47, 48, 50, 51, 54, 55, 56, 57, 58, 64, 68, 70, 72, 75, 78, 79, 85,
        86, 87, 89, 90, 91, 92, 95, 96, 97, 99, 103, 104
    ]

    # 3. 6가지 실험 조건 정의
    # Exp 01 (기본 설정): yolov8s, SGD, Scratch, epochs 100, batch 4, patience 20, box 10.0, cls 0.5, scale 0.25
    exp01 = {
        "name": "sgd_exp02_base",
        "optimizer": "SGD",
        "pretrained": False,
        "epochs": 100,
        "imgsz": 640,
        "batch": 4,
        "patience": 20,
        "lr0": 0.01,
        "lrf": 0.01,
        "cos_lr": False,
        "box": 10.0,
        "cls": 0.5,
        "dfl": 1.5,
        "scale": 0.25,
        "flipud": 0.0,
        "fliplr": 0.5,
        "hsv_s": 0.5,
        "hsv_v": 0.3,
        "degrees": 0.0,
        "translate": 0.1,
        "mosaic": 1.0,
        "close_mosaic": 10,
        "mixup": 0.0,
        "copy_paste": 0.0,
    }

    # Exp 02 (안정성 및 분류 가중치 증가): batch 16, patience 30, box 7.5, cls 1.0, scale 0.5
    exp02 = {
        "name": "sgd_exp03_stability",
        "optimizer": "SGD",
        "pretrained": False,
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,
        "patience": 30,
        "lr0": 0.01,
        "lrf": 0.01,
        "cos_lr": False,
        "box": 7.5,
        "cls": 1.0,
        "dfl": 1.5,
        "scale": 0.5,
        "flipud": 0.0,
        "fliplr": 0.5,
        "hsv_s": 0.5,
        "hsv_v": 0.3,
        "degrees": 0.0,
        "translate": 0.1,
        "mosaic": 1.0,
        "close_mosaic": 10,
        "mixup": 0.0,
        "copy_paste": 0.0,
    }

    # Exp 04 (배치 사이즈 변경): batch 8, patience 30, box 7.5, cls 1.0, scale 0.5
    exp04 = {
        "name": "sgd_exp04_batch8",
        "optimizer": "SGD",
        "pretrained": False,
        "epochs": 100,
        "imgsz": 640,
        "batch": 8,
        "patience": 30,
        "lr0": 0.01,
        "lrf": 0.01,
        "cos_lr": False,
        "box": 7.5,
        "cls": 1.0,
        "dfl": 1.5,
        "scale": 0.5,
        "flipud": 0.0,
        "fliplr": 0.5,
        "hsv_s": 0.5,
        "hsv_v": 0.3,
        "degrees": 0.0,
        "translate": 0.1,
        "mosaic": 1.0,
        "close_mosaic": 10,
        "mixup": 0.0,
        "copy_paste": 0.0,
    }

    # Exp 05 (학습률 스케줄 개선): Exp 02 기준, cos_lr True, lrf 0.001
    exp05 = {
        "name": "sgd_exp05_cos_lr",
        "optimizer": "SGD",
        "pretrained": False,
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,
        "patience": 30,
        "lr0": 0.01,
        "lrf": 0.001,
        "cos_lr": True,
        "box": 7.5,
        "cls": 1.0,
        "dfl": 1.5,
        "scale": 0.5,
        "flipud": 0.0,
        "fliplr": 0.5,
        "hsv_s": 0.5,
        "hsv_v": 0.3,
        "degrees": 0.0,
        "translate": 0.1,
        "mosaic": 1.0,
        "close_mosaic": 10,
        "mixup": 0.0,
        "copy_paste": 0.0,
    }

    # Exp 06 (Loss·증강 복합 개선): Exp 02 기준, cls 0.75, mixup 0.1, hsv_s 0.7, hsv_v 0.4, copy_paste 0.1
    exp06 = {
        "name": "sgd_exp06_imgsz960_aug_loss",
        "optimizer": "SGD",
        "pretrained": False,
        "epochs": 100,
        "imgsz": 960,
        "batch": 16,
        "patience": 30,
        "lr0": 0.01,
        "lrf": 0.01,
        "cos_lr": False,
        "box": 7.5,
        "cls": 0.75,
        "dfl": 1.5,
        "scale": 0.5,
        "flipud": 0.0,
        "fliplr": 0.5,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "mosaic": 1.0,
        "close_mosaic": 10,
        "mixup": 0.1,
        "copy_paste": 0.1,
    }

    # 전체 실험 목록 구성
    experiments = [exp06]

    # 4. 순차적인 학습 및 지정 클래스 대상 테스트 평가 반복문
    for exp in experiments:
        # 학습 및 Validation 수행 (Train Loss 확인용)
        train_and_val(exp, data_path=data_path, runs_dir=runs_dir, device=device)

        # 최고 성능 가중치 경로 정의
        best_weights = runs_dir / exp["name"] / "weights" / "best.pt"
        
        # Test 세트를 대상으로 51개 특정 클래스만 필터링하여 평가 실행
        evaluate_test_set(
            weights_path=best_weights,
            data_path=data_path,
            runs_dir=runs_dir,
            exp_name=exp["name"],
            batch_size=exp["batch"],
            device=device,
            target_classes=target_classes
        )