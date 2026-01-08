# /workspace/competition_xai/config/config.py

from dataclasses import dataclass, field


@dataclass
class Config:
    DEBUG_MODE: bool = False

    seed: int = 42
    run_dir: str = "/workspace/competition_xai/runs"
    datasets_path: str = "/workspace/competition_xai/datasets"
    train_dataset: str = "TIMIT"  #  KoDF | RVF | TIMIT
    model_name: str = "xception"
    input_modality: str = "rgb"  # rgb | wavelet | frequency | residual | npr
    mode: str = "evidence_harvesting"  # train | test | evidence_harvesting | llm_debate
    harvest_mode: str = "gradcam"  # gradcam | llm
    test_mode: str = "id"  # id | ood
    early_stopping_patience: int = 5
    early_stopping_delta: float = 1e-6
    ckpt_path: str | None = (  # 테스트나 다른거 할 때 체크포인트 꼭 확인할 것 !!!!
        "/workspace/competition_xai/runs/20260108_115555_xception_rgb_TIMIT"
    )
    use_gradcam: bool = True
    batch_size: int = 16
    num_epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-2
    img_size: int = 224
    num_workers: int = 0
    device: str = "cuda"

    pretrained: bool = True
    in_channels: int = 3
    num_classes: int = 2
    drop_rate: float = 0.0
    global_pool: str = "avg"

    key_json_path: str = "/workspace/competition_xai/keys.json"

    evidence_top_k: int = 200  # 라벨별로 몇개 저장할지
