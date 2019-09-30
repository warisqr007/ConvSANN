from enum import Enum
from models.cnn import CnnSiameseNet
from models.lstm import LSTMBasedSiameseNet
from models.multihead_attention import MultiheadAttentionSiameseNet
from models.bcann import AttentionCnn
from models.bcsann import AttentionSCnn
from models.bcsann_wmh import AttentionSCnnWMH
from models.twolayerbcnn import Attention2lyrCnn
from models.capsann import AttentionSCapnn
from models.capsnn import AttentionCapsnn
from models.sacnn_kb import AttentionSacnnKb


class ModelType(Enum):
    multihead = 0,
    rnn = 1,
    cnn = 2,
    bcann = 3,
    bcsann = 4,
    bcsannwmh = 5,
    twolayerbcnn=6,
    capsann = 7,
    capsnn = 8,
    sacnn_kb = 9


MODELS = {
    ModelType.cnn.name: CnnSiameseNet,
    ModelType.rnn.name: LSTMBasedSiameseNet,
    ModelType.multihead.name: MultiheadAttentionSiameseNet,
    ModelType.bcann.name: AttentionCnn,
    ModelType.bcsann.name: AttentionSCnn,
    ModelType.bcsannwmh.name: AttentionSCnnWMH,
    ModelType.twolayerbcnn.name:Attention2lyrCnn,
    ModelType.capsann.name:AttentionSCapnn,
    ModelType.capsnn.name:AttentionCapsnn,
    ModelType.sacnn_kb.name:AttentionSacnnKb
    
}

