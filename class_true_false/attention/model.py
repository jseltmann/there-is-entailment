import torch
import torch.nn as nn
import torch.nn.functional as F


class AttLSTM(nn.Module):

    def __init__(self, input_size=1, hidden_size=100, batch_size=64, max_len=25):
        super(AttLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.max_len = max_len
        
        self.caption_lstm = nn.LSTM(input_size,
                                    self.hidden_size,
                                    bidirectional=True)
        self.object_lstm = nn.LSTM(input_size,
                                   self.hidden_size,
                                   bidirectional=True)

        self.cap_attention = nn.Linear(4 * self.hidden_size, self.max_len)

        self.linear1 = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)
        self.linear2 = nn.Linear(2 * self.hidden_size, 2)

    def forward(self, caption, obj):
        cap_steps, (h_cap, _) = self.caption_lstm(caption)
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

        h_sum = torch.sum(att_applied, 0) / sent_len

        l1 = self.linear1(h_sum)
        out = self.linear2(l1)

        return out

    def reset_state(self):
        """
        Reset hidden states to zero.
        """
        self.hidden = (torch.zeros(1, self.batch_size, self.hidden_size),
                       torch.zeros(1, self.batch_size, self.hidden_size))



         
        


