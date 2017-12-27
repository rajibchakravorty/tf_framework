

from tf_train.trainer import Trainer
from cnn_1 import cnn_archi


import config

trainer = Trainer( cnn_archi, config )

trainer.train()