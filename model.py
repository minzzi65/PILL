from pathlib import Path
import torch
from ultralytics import YOLO


def train_and_test(exp_cfg: dict, data_path: Path, runs_dir: Path, device: str) -> None:
    exp_name = exp_cfg["name"]
    print("\n" + "=" * 60)
    print(f"Starting {exp_name}")
    print("=" * 60)

    model = YOLO("yolov8s.yaml")  # scratch
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

    best_model_path = runs_dir / exp_name / "weights" / "best.pt"
    if not best_model_path.exists():
        print(f"best.pt not found: {best_model_path}")
        return

    best_model = YOLO(str(best_model_path))
    metrics = best_model.val(
        data=str(data_path),
        split="test",
        batch=exp_cfg["batch"],
        device=device,
        workers=0,
        project=str(runs_dir),
        name=f"{exp_name}_test_eval",
        exist_ok=True,
    )

    print("-" * 60)
    print(f"{exp_name} Test Results")
    print(f"P         : {metrics.box.mp:.4f}")
    print(f"R         : {metrics.box.mr:.4f}")
    print(f"mAP50     : {metrics.box.map50:.4f}")
    print(f"mAP50-95  : {metrics.box.map:.4f}")
    print("-" * 60)


if __name__ == "__main__":
    data_path = Path(r"C:\Users\Administrator\pill_forder\data.yaml")
    runs_dir = Path(r"C:\Users\Administrator\pill_forder\exp_results")
    device = "0" if torch.cuda.is_available() else "cpu"

    print(f"Training device: {device}")
    print(f"Data config   : {data_path}")
    print(f"Results dir   : {runs_dir}")

    # Exp 01 (기본 설정): yolov8s, SGD, Scratch
    exp01 = {
        "name": "sgd_exp01_base",
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

    # Exp 02 (안정성 및 분류 가중치 증가)
    exp02 = {
        "name": "sgd_exp02_stability",
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

    # Exp 03 (학습 연장 및 색상/Mixup 증강)
    exp03 = {
        "name": "sgd_exp03_ext_aug",
        "optimizer": "SGD",
        "pretrained": False,
        "epochs": 130,
        "imgsz": 640,
        "batch": 4,
        "patience": 20,  # 별도 명시가 없으므로 Exp 01 기본값 준용
        "lr0": 0.01,
        "lrf": 0.01,
        "cos_lr": False,
        "box": 10.0,  # 별도 명시가 없으므로 Exp 01 기본값 준용
        "cls": 0.5,
        "dfl": 1.5,
        "scale": 0.25,
        "flipud": 0.0,
        "fliplr": 0.5,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "mosaic": 1.0,
        "close_mosaic": 10,
        "mixup": 0.05,
        "copy_paste": 0.0,
    }

    # Exp 04 (배치 사이즈 변경)
    # 기재해주신 조건이 Exp02와 유사하나 batch가 8인 세팅입니다.
    exp04 = {
        "name": "sgd_exp04_batch8",
        "optimizer": "SGD",
        "pretrained": False,
        "epochs": 100,
        "imgsz": 640,
        "batch": 8,
        "patience": 30,
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

    # Exp 05 (학습률 스케줄 개선): Exp 02 기준 유지
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

    # Exp 06 (Loss·증강 복합 개선): Exp 02 기준 유지
    exp06 = {
        "name": "sgd_exp06_complex",
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

    experiments = [exp01, exp02,  exp04]
    for exp in experiments:
        train_and_test(exp, data_path=data_path, runs_dir=runs_dir, device=device)