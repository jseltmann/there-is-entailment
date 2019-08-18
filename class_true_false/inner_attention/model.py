import torch
import torch.nn as nn
import torch.nn.functional as F


class InnerAttLSTM(nn.Module):

    def __init__(self, num_words=10000, emb_size=100, hidden_size=100, batch_size=64, max_len=25):
        super(InnerAttLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.max_len = max_len
        self.emb_size = emb_size
        
        self.embedding = nn.Embedding(num_words, self.emb_size)
        self.caption_lstm = nn.LSTM(self.emb_size,
                                    self.hidden_size,
                                    bidirectional=True)
        self.object_lstm = nn.LSTM(self.emb_size,
                                   self.hidden_size,
                                   bidirectional=True)

        #self.cap_attention = nn.Linear(4 * self.hidden_size, self.max_len)
        self.cap_attention = nn.Linear(self.max_len * self.hidden_size, self.max_len)
        self.pooling = nn.modules.pooling.AvgPool2d(3, stride=2)

        self.cap_linear = nn.Linear(2 * self.max_len * self.hidden_size, 2 * self.max_len * self.hidden_size)
        self.pool_linear = nn.Linear(12 * 24, 2 * self.max_len * self.hidden_size)
        # a bit hacky, but I don't quite know how the pooling reaches the new shape
        self.att_linear = nn.Linear(2 * self.max_len * self.hidden_size, 2 * self.max_len * self.hidden_size)
        self.att_softmax = nn.Softmax()

        self.linear1 = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)
        self.linear2 = nn.Linear(2 * self.hidden_size, 2)

    def forward(self, caption, obj):
        caption = caption.long()
        caption = self.embedding(caption)
        caption = torch.squeeze(caption)

        cap_steps, (h_cap, _) = self.caption_lstm(caption)
        cap_steps = torch.transpose(cap_steps, 1, 0)
        cap_pool = self.pooling(cap_steps)
        cap_pool = torch.flatten(cap_pool, start_dim=1)

        cap_steps_flat = torch.flatten(cap_steps, start_dim=1)
        cap_linear = self.cap_linear(cap_steps_flat)
        pool_linear = self.pool_linear(cap_pool)
        pool_linear = torch.reshape(pool_linear, (-1, self.max_len, 2 * self.hidden_size))
        eye = torch.eye(self.max_len, m=2*self.hidden_size)
        pool_linear = pool_linear * eye
        pool_linear = torch.flatten(pool_linear, start_dim=1)
        att_comb = cap_linear + pool_linear
        att_comb = torch.tanh(att_comb)
        att_linear = self.att_linear(att_comb)
        att_linear = self.att_softmax(att_linear)
        att_linear = torch.reshape(att_linear, (-1, self.max_len, 2*self.hidden_size))

        cap_steps_weighted = cap_steps * att_linear
        sent_len = cap_steps.size()[0]
        h_sum = torch.sum(cap_steps_weighted, 1) / sent_len
        #h_sum = torch.reshape(h_sum, (-1, 2, self.hidden_size))
        #print(h_sum.size())

        obj = obj.long()
        obj = self.embedding(obj)
        obj = obj.squeeze(2)
        _, (h_obj, _) = self.object_lstm(obj)

        # h_cap is tuple, since bidirectional lstm
        #h_sum = torch.stack((h_sum[0], h_sum[1]), dim=1)
        #h_sum = h_sum.reshape(-1, 2 * self.hidden_size)
        h_obj = torch.stack((h_obj[0], h_obj[1]), dim=1)
        h_obj = h_obj.reshape(-1, 2 * self.hidden_size) # (BATCH_SIZE * HIDDEN_SIZE)

        h_combined = torch.cat([h_sum, h_obj], 1)

        #att_weights = self.cap_attention(h_combined)
        #att_weights = torch.transpose(att_weights, 0, 1)
        #att_weights = torch.unsqueeze(att_weights, 2)
        #sent_len = cap_steps.size()[0]
        #att_weights = att_weights[sent_len-1,:,:]
        #att_applied = att_weights * cap_steps

        #h_sum = torch.sum(att_applied, 0)

        #l1 = self.linear1(h_sum)
        l1 = self.linear1(h_obj)
        out = self.linear2(l1)

        return out

    def reset_state(self):
        """
        Reset hidden states to zero.
        """
        self.hidden = (torch.zeros(1, self.batch_size, self.hidden_size),
                       torch.zeros(1, self.batch_size, self.hidden_size))



         
        


