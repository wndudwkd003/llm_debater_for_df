# /workspace/competition_xai/run.py

from dataclasses import dataclass, field

from competition_xai.train import Trainer
from competition_xai.llm import LLMDebater

from competition_xai.utils.seed_utils import set_seeds
from competition_xai.analysis import Analyzer


@dataclass
class Config:
    seed: int = 42
    run_dir: str = "/workspace/competition_xai/runs"
    datasets_path: str = "/workspace/competition_xai/datasets"
    train_dataset: str = "KoDF"
    model_name: str = "xception"
    input_modality: str = "rgb"  # rgb | wavelet | frequency | residual | npr
    mode: str = "train"  # train | test | evidence_harvesting | llm_debate
    test_mode: str = "ood"  # iod | ood
    ckpt_path: str = ""
    use_gradcam: bool = True
    batch_size: int = 16
    num_epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-2
    img_size: int = 224
    num_workers: int = 0
    device: str = "cuda"


def main(config: Config):
    if config.mode == "train":
        results = Trainer(config).train()
        Analyzer(config, results).train_generate_reports()

    elif config.mode == "test":
        results = Trainer(config).test()
        Analyzer(config, results).test_generate_reports()

    elif config.mode == "evidence_harvesting":
        results = LLMDebater(config).harvest_evidence()
        Analyzer(config, results).evidence_harvesting_generate_reports()

    elif config.mode == "llm_debate":
        results = LLMDebater(config).llm_debate()
        Analyzer(config, results).llm_debate_generate_reports()


if __name__ == "__main__":
    config = Config()
    set_seeds(config.seed)
    main(config)
