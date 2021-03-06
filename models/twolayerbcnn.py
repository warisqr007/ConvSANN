from layers.convolution import cnn_layers
from layers.losses import mse
from layers.similarity import manhattan_similarity
from layers.attention import stacked_multihead_attention
from models.base_model import BaseSiameseNet
from utils.config_helpers import parse_list
from layers.basics import dropout
import tensorflow as tf
import numpy as np

_conv_projection_size = 64
_attention_output_size = 200
_comparison_output_size = 150

class Attention2lyrCnn(BaseSiameseNet):

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
            pad = tf.zeros([tf.shape(self.x1)[0], 1, _conv_projection_size])
            return tf.concat([pad, values, pad], axis=1)
        

    def attention_layer0(self,sent1,sent2,sent1_len,sent2_len):
        with tf.name_scope('attention_layer'):
            #e_X1 = tf.layers.dense(sent1, attention_output_size, activation=tf.nn.relu, name='attention_nn')
            #e_X2 = tf.layers.dense(sent2, attention_output_size, activation=tf.nn.relu, name='attention_nn', reuse=True)
            W=tf.get_variable("W11", shape=(tf.to_int32(tf.shape(self.x1)[0]),64,64), initializer=tf.random_normal_initializer(),dtype=tf.float32)
      
            h=tf.matmul(sent1,W)
            #print(h)
            e = tf.matmul(h, sent2, transpose_b=True, name='e1')
            #print(e)
            beta = tf.matmul(self._masked_softmax(e, sent2_len), sent2, name='beta1')
            #print(beta)
            alpha = tf.matmul(self._masked_softmax(tf.transpose(e, [0,2,1]), sent1_len), sent1, name='alpha1')
            #print(alpha)
            return (alpha,beta)
            
    def siamese_layer(self, sequence_len, model_cfg):
        _conv_filter_size = 3
        #parse_list(model_cfg['PARAMS']['filter_sizes'])
        with tf.name_scope('convolutional_layer'):
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
            
            X1_conv_12 = tf.layers.conv1d(
                self._conv_pad(self._X1_conv),
                _conv_projection_size,
                _conv_filter_size,
                padding='valid',
                use_bias=False,
                name='conv_12',
            )
            
            X2_conv_12 = tf.layers.conv1d(
                self._conv_pad(self._X2_conv),
                _conv_projection_size,
                _conv_filter_size,
                padding='valid',
                use_bias=False,
                name='conv_12',
                reuse=True
            )
            
            X1_conv_12 = tf.layers.dropout(X1_conv_12, rate=self.dropout, training=self.is_training)
            X2_conv_12 = tf.layers.dropout(X2_conv_12, rate=self.dropout, training=self.is_training)
            
            X1_conv_22 = tf.layers.conv1d(
                self._conv_pad(X1_conv_12),
                _conv_projection_size,
                _conv_filter_size,
                padding='valid',
                use_bias=False,
                name='conv_22',
            )
            
            X2_conv_22 = tf.layers.conv1d(
                self._conv_pad(X2_conv_12),
                _conv_projection_size,
                _conv_filter_size,
                padding='valid',
                use_bias=False,
                name='conv_22',
                reuse=True
            )
            
            self._X1_conv2 = tf.layers.dropout(X1_conv_22, rate=self.dropout, training=self.is_training)
            self._X2_conv2 = tf.layers.dropout(X2_conv_22, rate=self.dropout, training=self.is_training)
            
        with tf.name_scope('attention_layer'):
            e_X1 = tf.layers.dense(self._X1_conv, _attention_output_size, activation=tf.nn.relu, name='attention_nn')
            
            e_X2 = tf.layers.dense(self._X2_conv, _attention_output_size, activation=tf.nn.relu, name='attention_nn', reuse=True)
            
            e = tf.matmul(e_X1, e_X2, transpose_b=True, name='e')
            
            self._beta = tf.matmul(self._masked_softmax(e, sequence_len), self._X2_conv, name='beta2')
            self._alpha = tf.matmul(self._masked_softmax(tf.transpose(e, [0,2,1]), sequence_len), self._X1_conv, name='alpha2')
            
        with tf.name_scope('attention_layer_conv2'):
            e_X1 = tf.layers.dense(self._X1_conv2, _attention_output_size, activation=tf.nn.relu, name='attention_nn_1')
            
            e_X2 = tf.layers.dense(self._X2_conv2, _attention_output_size, activation=tf.nn.relu, name='attention_nn_1', reuse=True)
            
            e = tf.matmul(e_X1, e_X2, transpose_b=True, name='e')
            
            self._beta_lr2 = tf.matmul(self._masked_softmax(e, sequence_len), self._X2_conv2, name='betalr2')
            self._alpha_lr2 = tf.matmul(self._masked_softmax(tf.transpose(e, [0,2,1]), sequence_len), self._X1_conv2, name='alphalr2')
            
        with tf.name_scope('self_attention1'):
            e_X1 = tf.layers.dense(self._X1_conv, _attention_output_size, activation=tf.nn.relu, name='attention_nn1')
            
            e_X2 = tf.layers.dense(self._X1_conv, _attention_output_size, activation=tf.nn.relu, name='attention_nn1', reuse=True)
            
            e = tf.matmul(e_X1, e_X2, transpose_b=True, name='e')
            
            self._beta1 = tf.matmul(self._masked_softmax(e, sequence_len), self._X1_conv, name='betaslf1') 
            
        with tf.name_scope('self_attention1_conv2'):
            e_X1 = tf.layers.dense(self._X1_conv2, _attention_output_size, activation=tf.nn.relu, name='attention_nn2')
            
            e_X2 = tf.layers.dense(self._X1_conv2, _attention_output_size, activation=tf.nn.relu, name='attention_nn2', reuse=True)
            
            e = tf.matmul(e_X1, e_X2, transpose_b=True, name='e')
            
            self._beta1_lr2 = tf.matmul(self._masked_softmax(e, sequence_len), self._X1_conv2, name='betaslf2') 
            
        with tf.name_scope('self_attention2'):
            e_X1 = tf.layers.dense(self._X2_conv, _attention_output_size, activation=tf.nn.relu, name='attention_nn3')
            
            e_X2 = tf.layers.dense(self._X2_conv, _attention_output_size, activation=tf.nn.relu, name='attention_nn3', reuse=True)
            
            e = tf.matmul(e_X1, e_X2, transpose_b=True, name='e')
            
            self._alpha1 = tf.matmul(self._masked_softmax(e, sequence_len), self._X2_conv, name='betaslf3')
            
        with tf.name_scope('self_attention2_conv2'):
            e_X1 = tf.layers.dense(self._X2_conv2, _attention_output_size, activation=tf.nn.relu, name='attention_nn4')
            
            e_X2 = tf.layers.dense(self._X2_conv2, _attention_output_size, activation=tf.nn.relu, name='attention_nn4', reuse=True)
            
            e = tf.matmul(e_X1, e_X2, transpose_b=True, name='e')
            
            self._alpha1_lr2 = tf.matmul(self._masked_softmax(e, sequence_len), self._X2_conv2, name='betaslf4')
          
            
        with tf.name_scope('comparison_layer'):
            X1_comp = tf.layers.dense(
                tf.concat([self._X1_conv, self._beta, self._beta1], 2),
                _comparison_output_size,
                activation=tf.nn.relu,
                name='comparison_nn'
            )
            self._X1_comp = tf.multiply(
                tf.layers.dropout(X1_comp, rate=self.dropout, training=self.is_training),
                tf.expand_dims(tf.sequence_mask(sequence_len, tf.reduce_max(sequence_len), dtype=tf.float32), -1)
            )
            
            X1_complr2 = tf.layers.dense(
                tf.concat([self._X1_conv2, self._beta_lr2, self._beta1_lr2], 2),
                _comparison_output_size,
                activation=tf.nn.relu,
                name='comparison_nn_lr2'
            )
            self._X1_complr2 = tf.multiply(
                tf.layers.dropout(X1_complr2, rate=self.dropout, training=self.is_training),
                tf.expand_dims(tf.sequence_mask(sequence_len, tf.reduce_max(sequence_len), dtype=tf.float32), -1)
            )
            
            X2_comp = tf.layers.dense(
                tf.concat([self._X2_conv, self._alpha,self._alpha1], 2),
                _comparison_output_size,
                activation=tf.nn.relu,
                name='comparison_nn',
                reuse=True
            )
            self._X2_comp = tf.multiply(
                tf.layers.dropout(X2_comp, rate=self.dropout, training=self.is_training),
                tf.expand_dims(tf.sequence_mask(sequence_len, tf.reduce_max(sequence_len), dtype=tf.float32), -1)
            )
            
            X2_complr2 = tf.layers.dense(
                tf.concat([self._X2_conv2, self._alpha_lr2,self._alpha1_lr2], 2),
                _comparison_output_size,
                activation=tf.nn.relu,
                name='comparison_nn_lr2',
                reuse=True
            )
            self._X2_complr2 = tf.multiply(
                tf.layers.dropout(X2_complr2, rate=self.dropout, training=self.is_training),
                tf.expand_dims(tf.sequence_mask(sequence_len, tf.reduce_max(sequence_len), dtype=tf.float32), -1)
            )
        
            X1_agg = tf.reduce_sum(self._X1_comp, 1)
            X2_agg = tf.reduce_sum(self._X2_comp, 1)
            X1_agglr2 =tf.reduce_sum(self._X1_complr2, 1)
            X2_agglr2 =tf.reduce_sum(self._X2_complr2, 1)
            _agg1 = tf.concat([X1_agg, X1_agglr2], 1)
            _agg2 = tf.concat([X2_agg, X2_agglr2],1)
        
        return manhattan_similarity(_agg1,_agg2)
