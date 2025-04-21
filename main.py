from src.evalution import evalutate
from src.utils.helper import log
from src.models import transformers_trainer


if __name__ == '__main__':
    transformers_trainer.run()
    evalutate.run()