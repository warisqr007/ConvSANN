from enum import Enum
from models.cnn import CnnSiameseNet
from models.lstm import LSTMBasedSiameseNet
from models.multihead_attention import MultiheadAttentionSiameseNet
from models.bcann import AttentionCnn


class ModelType(Enum):
    multihead = 0,
    rnn = 1,
    cnn = 2,
    bcann = 3,
    bcsann = 4


MODELS = {
    ModelType.cnn.name: CnnSiameseNet,
    ModelType.rnn.name: LSTMBasedSiameseNet,
    ModelType.multihead.name: MultiheadAttentionSiameseNet,
    ModelType.bcann.name: AttentionCnn,
    ModelType.bcsann.name: AttentionSCnn
}

