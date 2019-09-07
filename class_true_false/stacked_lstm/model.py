import torch
import torch.nn as nn
import torch.nn.functional as F


class StackedLSTM(nn.Module):

    def __init__(self, num_words=10000, emb_size=100, hidden_size=100, batch_size=64, max_len=25):
        super(StackedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.max_len = max_len
        self.emb_size = emb_size
        
        self.embedding = nn.Embedding(num_words, self.emb_size)
        self.caption_lstm1 = nn.LSTM(self.emb_size,
                                    self.hidden_size,
                                    bidirectional=True)
        self.caption_lstm2 = nn.LSTM(self.hidden_size * 2,
                                    self.hidden_size,
                                    bidirectional=True)
        self.caption_lstm3 = nn.LSTM(self.hidden_size * 2,
                                    self.hidden_size,
                                    bidirectional=True)
        self.object_lstm = nn.LSTM(self.emb_size,
                                   self.hidden_size,
                                   bidirectional=True)

        self.cap_attention = nn.Linear(4 * self.hidden_size, self.max_len)

        self.linear1 = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)
        self.linear2 = nn.Linear(2 * self.hidden_size, 2)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, caption, obj):
        caption = caption.long()
        caption = self.embedding(caption)
        caption = torch.squeeze(caption)

        cap_steps, (h_cap, _) = self.caption_lstm1(caption)
        cap_steps, (h_cap, _) = self.caption_lstm2(cap_steps)
        cap_steps, (h_cap, _) = self.caption_lstm3(cap_steps)
        

        obj = obj.long()
        obj = self.embedding(obj)
        obj = obj.squeeze(2)
        _, (h_obj, _) = self.object_lstm(obj)

        # h_cap is tuple, since bidirectional lstm
        h_cap = torch.stack((h_cap[0], h_cap[1]), dim=1)
        h_cap = h_cap.reshape(-1, 2 * self.hidden_size)
        h_obj = torch.stack((h_obj[0], h_obj[1]), dim=1)
        h_obj = h_obj.reshape(-1, 2 * self.hidden_size) # (BATCH_SIZE * HIDDEN_SIZE)

        h_combined = torch.cat([h_cap, h_obj], 1)

        att_weights = self.cap_attention(h_combined)
        att_weights = torch.transpose(att_weights, 0, 1)
        att_weights = torch.unsqueeze(att_weights, 2)
        sent_len = cap_steps.size()[0]
        att_weights = att_weights[sent_len-1,:,:]
        att_applied = att_weights * cap_steps

        h_sum = torch.sum(att_applied, 0)

        l1 = torch.tanh(self.linear1(h_sum))
        l2 = self.linear2(l1)
        out = self.softmax(l2)

        return out

    def reset_state(self):
        """
        Reset hidden states to zero.
        """
        self.hidden = (torch.zeros(1, self.batch_size, self.hidden_size),
                       torch.zeros(1, self.batch_size, self.hidden_size))



         
        


