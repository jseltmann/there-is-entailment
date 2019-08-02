import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLSTM(nn.Module):

    def __init__(self, input_size=1, hidden_size=100, batch_size=64):
        super(BaseLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        
        self.caption_lstm = nn.LSTM(input_size,
                                    self.hidden_size,
                                    bidirectional=True)
        self.object_lstm = nn.LSTM(input_size,
                                   self.hidden_size,
                                   bidirectional=True)

        self.linear = nn.Linear(4 * self.hidden_size, 2)

    def forward(self, caption, obj):
        _, (h_cap, _) = self.caption_lstm(caption)
        _, (h_obj, _) = self.object_lstm(obj)

        # h_cap is tuple, since bidirectional lstm
        h_cap = torch.stack((h_cap[0], h_cap[1]), dim=1)
        h_cap = h_cap.reshape(-1, 2 * self.hidden_size)
        h_obj = torch.stack((h_obj[0], h_obj[1]), dim=1)
        h_obj = h_obj.reshape(-1, 2 * self.hidden_size) # (BATCH_SIZE * HIDDEN_SIZE)

        h_combined = torch.cat([h_cap, h_obj], 1)

        out = self.linear(h_combined)

        return out

    def reset_state(self):
        """
        Reset hidden states to zero.
        """
        self.hidden = (torch.zeros(1, self.batch_size, self.hidden_size),
                       torch.zeros(1, self.batch_size, self.hidden_size))



         
        


