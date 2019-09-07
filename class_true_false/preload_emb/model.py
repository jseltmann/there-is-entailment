import torch
import torch.nn as nn
import torch.nn.functional as F



class PreloadEmbLSTM(nn.Module):

    def __init__(self, num_words=10000, emb_size=100, 
                 hidden_size=100, batch_size=64, 
                 emb_file=None, word2num=None):
        super(PreloadEmbLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.batch_size = batch_size
        
        if not emb_file is None and not word2num is None:
            vectors = load_vectors(emb_file, word2num)
            self.embedding = nn.Embedding.from_pretrained(vectors)
        else:
            self.embedding = nn.Embedding(num_words, self.emb_size)
        self.caption_lstm = nn.LSTM(self.emb_size,
                                    self.hidden_size,
                                    bidirectional=True)
        self.object_lstm = nn.LSTM(self.emb_size,
                                   self.hidden_size,
                                   bidirectional=True)

        self.linear1 = nn.Linear(4 * self.hidden_size, 4 * self.hidden_size)
        self.linear2 = nn.Linear(4 * self.hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, caption, obj):
        caption = caption.long()
        caption = self.embedding(caption)
        caption = torch.squeeze(caption)
        _, (h_cap, _) = self.caption_lstm(caption)

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

        l1 = torch.tanh(self.linear1(h_combined))
        out = self.softmax(self.linear2(l1))

        return out

    def reset_state(self):
        """
        Reset hidden states to zero.
        """
        self.hidden = (torch.zeros(1, self.batch_size, self.hidden_size),
                       torch.zeros(1, self.batch_size, self.hidden_size))



         
        


def load_vectors(vector_filename, word2num):
    """
    Load glove vectors.

    Parameters
    ----------
    vector_filename : str
        File to read the vectors from.
    word2num : dict[str]
        Mapping from words to indices.

    Return
    ------
    vectors : dict[str]
        Dict containing the vector for each word.
    """

    with open(vector_filename) as vf:
        for line in vf:
            entries = line.split()
            emb_dim = len(entries) - 1
            break

    vectors = torch.zeros(len(word2num), emb_dim)

    with open(vector_filename) as vector_file:
        for i, line in enumerate(vector_file):
            entries = line.split()
            word = entries[0].lower()
            if word in word2num:
                nums = entries[1:]
                nums = [float(num) for num in nums]
                vector = torch.Tensor(nums)
                position = word2num[word]
                vectors[position] = vector

    return vectors
