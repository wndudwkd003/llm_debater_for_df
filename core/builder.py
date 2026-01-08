# /workspace/competition_xai/core/builder.py

from config.config import Config


class ModelBuilder:
    @staticmethod
    def build(config: Config):

        if config.model_name == "xception":
            from core.backbone.xception import build_model

        else:
            raise ValueError("[builder.py] 모델 이름이 틀렸음")

        model = build_model(config)
        return model
