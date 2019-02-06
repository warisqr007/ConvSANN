from keras.layers import LeakyReLU, Dense, Input, Embedding, Dropout, Bidirectional, GRU, Flatten, SpatialDropout1D
from layers.convolution import cnn_layers
from layers.losses import mse
from layers.similarity import manhattan_similarity
from models.base_model import BaseSiameseNet
from models.Capsule_Keras import *
from utils.config_helpers import parse_list
from layers.basics import dropout
import tensorflow as tf
import numpy as np

_conv_projection_size = 64
_attention_output_size = 200
_comparison_output_size = 120

gru_len = 256
Routings = 3
Num_capsule = 10
Dim_capsule = 16
dropout_p = 0.25
rate_drop_dense = 0.28

class AttentionCapsnn(BaseSiameseNet):

    def __init__(self, max_sequence_len, vocabulary_size, main_cfg, model_cfg):
        BaseSiameseNet.__init__(self, max_sequence_len, vocabulary_size, main_cfg, model_cfg, mse)
          
        
    def _masked_softmax(self, values, lengths):
        with tf.name_scope('MaskedSoftmax'):
            mask = tf.expand_dims(tf.sequence_mask(lengths, tf.reduce_max(lengths), dtype=tf.float32), -2)
    
            inf_mask = (1 - mask) * -np.inf
            inf_mask = tf.where(tf.is_nan(inf_mask), tf.zeros_like(inf_mask), inf_mask)

            return tf.nn.softmax(tf.multiply(values, mask) + inf_mask)
        
    def _conv_pad(self, values):
        with tf.name_scope('convolutional_padding'):
            pad = tf.zeros([tf.shape(self.x1)[0], 1, self.embedding_size])
            return tf.concat([pad, values, pad], axis=1)
        

    def _attention_layer(self):
        with tf.name_scope('attention_layer'):
            e_X1 = tf.layers.dense(self._X1_conv, _attention_output_size, activation=tf.nn.relu, name='attention_nn')
            e_X2=tf.layers.dense(self._X2_conv, _attention_output_size, activation=tf.nn.relu,
                                 name='attention_nn',reuse=True)
            
            e = tf.matmul(e_X1, e_X2, transpose_b=True, name='e')
            
            self._beta = tf.matmul(self._masked_softmax(e, sequence_len), self._X2_conv, name='beta2')
            self._alpha = tf.matmul(self._masked_softmax(tf.transpose(e, [0,2,1]), sequence_len), self._X1_conv, name='alpha2')
            
    def siamese_layer(self, sequence_len, model_cfg):
        _conv_filter_size = 3
        #parse_list(model_cfg['PARAMS']['filter_sizes'])
        with tf.name_scope('capsule_layer'): 
            #self.embedded_x1 = self.embedded_x1[...,tf.newaxis] 
            #self.embedded_x2 = self.embedded_x2[...,tf.newaxis]
            #self._X1_caps, activations1 = capsule_model_B(self.embedded_x1, 3)
            #self._X2_caps, activations2 = capsule_model_B(self.embedded_x2, 3)
            
            x1 = Bidirectional(GRU(gru_len,
                          activation='relu',
                          dropout=dropout_p,
                          recurrent_dropout=dropout_p,
                          return_sequences=True))(self.embedded_x1)
            self._X1_caps = Capsule(num_capsule=Num_capsule,
                                    dim_capsule=Dim_capsule,
                                    routings=Routings,
                                    share_weights=True)(x1)

            #self._X1_caps = Flatten()(self._X1_caps)
            #self._X1_caps = Dropout(dropout_p)(self._X1_caps)
            #self._X1_caps = LeakyReLU()(self._X1_caps)
            
            x2 = Bidirectional(GRU(gru_len,
                          activation='relu',
                          dropout=dropout_p,
                          recurrent_dropout=dropout_p,
                          return_sequences=True))(self.embedded_x2)
            self._X2_caps = Capsule(num_capsule=Num_capsule,
                                    dim_capsule=Dim_capsule,
                                    routings=Routings,
                                    share_weights=True)(x2)
            
            '''
            X1_conv_1 = tf.layers.conv1d(
                self._conv_pad(self.embedded_x1),
                _conv_projection_size,
                _conv_filter_size,
                padding='valid',
                use_bias=False,
                name='conv_1',
            )
            
            X2_conv_1 = tf.layers.conv1d(
                self._conv_pad(self.embedded_x2),
                _conv_projection_size,
                _conv_filter_size,
                padding='valid',
                use_bias=False,
                name='conv_1',
                reuse=True
            )
            
            X1_conv_1 = tf.layers.dropout(X1_conv_1, rate=self.dropout, training=self.is_training)
            X2_conv_1 = tf.layers.dropout(X2_conv_1, rate=self.dropout, training=self.is_training)
            
            X1_conv_2 = tf.layers.conv1d(
                self._conv_pad(X1_conv_1),
                _conv_projection_size,
                _conv_filter_size,
                padding='valid',
                use_bias=False,
                name='conv_2',
            )
            
            X2_conv_2 = tf.layers.conv1d(
                self._conv_pad(X2_conv_1),
                _conv_projection_size,
                _conv_filter_size,
                padding='valid',
                use_bias=False,
                name='conv_2',
                reuse=True
            )
            
            self._X1_conv = tf.layers.dropout(X1_conv_2, rate=self.dropout, training=self.is_training)
            self._X2_conv = tf.layers.dropout(X2_conv_2, rate=self.dropout, training=self.is_training)
            '''
            
        with tf.name_scope('attention_layer'):
            e_X1 = tf.layers.dense(self._X1_caps, _attention_output_size, activation=tf.nn.relu, name='attention_nn')
            
            e_X2 = tf.layers.dense(self._X2_caps, _attention_output_size, activation=tf.nn.relu, name='attention_nn', reuse=True)
            
            e = tf.matmul(e_X1, e_X2, transpose_b=True, name='e')
            
            self._beta = tf.matmul(self._masked_softmax(e, sequence_len), self._X2_conv, name='beta2')
            self._alpha = tf.matmul(self._masked_softmax(tf.transpose(e, [0,2,1]), sequence_len), self._X1_conv, name='alpha2')
            
        with tf.name_scope('self_attention1'):
            e_X1 = tf.layers.dense(self._X1_caps, _attention_output_size, activation=tf.nn.relu, name='attention_nn1')
            
            e_X2 = tf.layers.dense(self._X1_caps, _attention_output_size, activation=tf.nn.relu, name='attention_nn1', reuse=True)
            
            e = tf.matmul(e_X1, e_X2, transpose_b=True, name='e')
            
            self._beta1 = tf.matmul(self._masked_softmax(e, sequence_len), self._X1_conv, name='beta2') 
            
        with tf.name_scope('self_attention2'):
            e_X1 = tf.layers.dense(self._X2_caps, _attention_output_size, activation=tf.nn.relu, name='attention_nn2')
            
            e_X2 = tf.layers.dense(self._X2_caps, _attention_output_size, activation=tf.nn.relu, name='attention_nn2', reuse=True)
            
            e = tf.matmul(e_X1, e_X2, transpose_b=True, name='e')
            
            self._alpha1 = tf.matmul(self._masked_softmax(e, sequence_len), self._X2_conv, name='beta2')
            
        with tf.name_scope('comparison_layer'):
            X1_comp = tf.layers.dense(
                tf.concat([self._X1_caps, self._beta, self._beta1], 2),
                _comparison_output_size,
                activation=tf.nn.relu,
                name='comparison_nn'
            )
            self._X1_comp = tf.multiply(
                tf.layers.dropout(X1_comp, rate=self.dropout, training=self.is_training),
                tf.expand_dims(tf.sequence_mask(sequence_len, tf.reduce_max(sequence_len), dtype=tf.float32), -1)
            )
            
            X2_comp = tf.layers.dense(
                tf.concat([self._X2_caps, self._alpha,self._alpha1], 2),
                _comparison_output_size,
                activation=tf.nn.relu,
                name='comparison_nn',
                reuse=True
            )
            self._X2_comp = tf.multiply(
                tf.layers.dropout(X2_comp, rate=self.dropout, training=self.is_training),
                tf.expand_dims(tf.sequence_mask(sequence_len, tf.reduce_max(sequence_len), dtype=tf.float32), -1)
            )
        
            X1_agg = tf.reduce_sum(self._X1_comp, 1)
            X2_agg = tf.reduce_sum(self._X2_comp, 1)
            
            self._agg = tf.concat([X1_agg, X2_agg], 1)
        
        return manhattan_similarity(X1_agg,X2_agg)
