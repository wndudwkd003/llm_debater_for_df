# /workspace/competition_xai/run.py

from config.config import Config
from worker.train import Trainer
from worker.llm import LLMDebater

from utils.seed_utils import set_seeds
from worker.analysis import Analyzer
import os
import json


def get_in_channels(im: str) -> int:
    if im == "rgb":
        return 3
    else:
        return 1


def api_register(config: Config):
    key_json_path = config.key_json_path

    with open(key_json_path, "r") as f:
        keys = json.load(f)

    for k, v in keys.items():
        os.environ[k] = v


def main(config: Config):
    print(f"[run.py] Mode: {config.mode}")
    print(config)
    print("-" * 50)
    print()

    # 입력 채널 설정
    config.in_channels = get_in_channels(config.input_modality)

    if config.mode == "train":
        trainer = Trainer(config)
        results = trainer.train()
        Analyzer(config, results).train_generate_reports()

    elif config.mode == "test":
        trainer = Trainer(config)
        results = trainer.test()
        Analyzer(config, results).test_generate_reports()

    elif config.mode == "evidence_harvesting":
        debater = LLMDebater(config)
        if config.harvest_mode == "gradcam":
            results = debater.harvest_evidence()

        elif config.harvest_mode == "llm":
            results = debater.call_llm_on_saved_evidence()

        Analyzer(config, results).evidence_harvesting_generate_reports()

    elif config.mode == "llm_debate":
        debater = LLMDebater(config)
        results = debater.llm_debate()
        Analyzer(config, results).llm_debate_generate_reports()

    print(f"[run.py] Finished {config.mode}.")


"""

keys.json을 만드셨나요?

"""


if __name__ == "__main__":
    config = Config()
    set_seeds(config.seed)
    api_register(config)
    main(config)
