#from src.evalution import evalutate
#from src.utils.helper import log
#from src.models import transformers_trainer

from src.data import bert_load_data
from src.models import bert_trainer
from src.models import bert_predict


if __name__ == '__main__':
    #transformers_trainer.run()
    #evalutate.run()
    [train_dataset, test_dataset] = bert_load_data.run()
    trainer = bert_trainer.customTraining(train_dataset, test_dataset)
    bert_predict.custom_get_classification_report(trainer, test_dataset)
