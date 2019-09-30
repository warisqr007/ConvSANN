
class BatchHelper:

    def __init__(self,  x1, x2, labels, batch_size, max_sentence_len, kb_dict):
        self.x1 = x1
        # self.x1 = self.x1.reshape(-1, 1)
        self.x2 = x2
        # self.x2 = self.x2.reshape(-1, 1)
        self.labels = labels
        self.labels = self.labels.reshape(-1, 1)
        self.batch_size = batch_size
        self.maxlen_y = 32
        self.maxlen_x = 32

    def next(self, batch_id):
        x1_batch = self.x1[batch_id * self.batch_size:(batch_id + 1) * self.batch_size]
        x2_batch = self.x2[batch_id * self.batch_size:(batch_id + 1) * self.batch_size]
        labels_batch = self.labels[batch_id * self.batch_size:(batch_id + 1) * self.batch_size]
        
        kb_x = numpy.zeros((self.maxlen_x, self.batch_size, self.maxlen_y, 5)).astype('float32')
        kb_y = numpy.zeros((self.maxlen_y, self.batch_size, self.maxlen_x, 5)).astype('float32')
        kb_att = numpy.zeros((self.maxlen_x, self.batch_size, self.maxlen_y)).astype('float32')
    
        for idx, [s_xl, s_yl, ll] in enumerate(zip(x1_batch, x2_batch, labels_batch)):
            for sid, s in enumerate(s_xl):
                for tid, t in enumerate(s_yl):
                    if s in kb_dict:
                        if t in kb_dict[s]:
                            kb_x[sid, idx, tid, :] = numpy.array(kb_dict[s][t]).astype('float32')
                            kb_att[sid, idx, tid] = 1.

            for sid, s in enumerate(s_yl):
                for tid, t in enumerate(s_xl):
                    if s in kb_dict:
                        if t in kb_dict[s]:
                            kb_y[sid, idx, tid, :] = numpy.array(kb_dict[s][t]).astype('float32')


        #return x, x_mask, kb_x, y, y_mask, kb_y, kb_att, l
        return x1_batch, x2_batch, labels_batch, kb_x, kb_y, kb_att
