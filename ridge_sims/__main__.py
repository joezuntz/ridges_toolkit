from .steps import step1, step2, step3
from .config import Config


if __name__ == "__main__":
    config = Config()
    config.save()
    step1(config)
    step2(config)
    step3(config)
