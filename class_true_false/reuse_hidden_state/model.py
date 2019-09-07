import torch
import torch.nn as nn
import torch.nn.functional as F


class ReuseStateLSTM(nn.Module):

    def __init__(self, num_words=10000, emb_size=100, hidden_size=100, batch_size=64):
        super(ReuseStateLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.batch_size = batch_size
        
        self.embedding = nn.Embedding(num_words, self.emb_size)
        self.caption_lstm = nn.LSTM(self.emb_size,
                                    self.hidden_size,
                                    bidirectional=True)
        self.object_lstm = nn.LSTM(self.emb_size,
                                   self.hidden_size,
                                   bidirectional=True)

        self.linear1 = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)
        self.linear2 = nn.Linear(2 * self.hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, caption, obj):
        obj = obj.long()
        obj = self.embedding(obj)
        obj = obj.squeeze(2)
        _, (h_obj, c_obj) = self.object_lstm(obj)

        caption = caption.long()
        caption = self.embedding(caption)
        caption = torch.squeeze(caption)
        _, (h_cap, c_cap) = self.caption_lstm(caption, (h_obj,c_obj))

        # h_cap is tuple, since bidirectional lstm
        h_cap = torch.stack((h_cap[0], h_cap[1]), dim=1)
        h_cap = h_cap.reshape(-1, 2 * self.hidden_size)
        #h_obj = torch.stack((h_obj[0], h_obj[1]), dim=1)
        #h_obj = h_obj.reshape(-1, 2 * self.hidden_size) # (BATCH_SIZE * HIDDEN_SIZE)

        #h_combined = torch.cat([h_cap, h_obj], 1)

        l1 = torch.tanh(self.linear1(h_cap))
        out = self.softmax(self.linear2(l1))

        return out

    def reset_state(self):
        """
        Reset hidden states to zero.
        """
        self.hidden = (torch.zeros(1, self.batch_size, self.hidden_size),
                       torch.zeros(1, self.batch_size, self.hidden_size))



         
        


