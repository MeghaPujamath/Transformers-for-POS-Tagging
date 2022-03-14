from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import POSTaggingModel

import random
torch.manual_seed(0)
random.seed(0)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, n_head=4, d_qkv=32, dropout=0.1, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_qkv = d_qkv

        # We provide these model parameters to give an example of a weight
        # initialization approach that we know works well for our tasks. Feel free
        # to delete these lines and write your own, if you would like.
        self.w_q = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
        self.w_k = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
        self.w_v = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
        self.w_o = nn.Parameter(torch.Tensor(n_head, d_qkv, d_model))
        nn.init.xavier_normal_(self.w_q)
        nn.init.xavier_normal_(self.w_k)
        nn.init.xavier_normal_(self.w_v)
        nn.init.xavier_normal_(self.w_o)

        self.fc_out = nn.Linear(n_head * self.d_qkv, d_model)

        # self.w_k = nn.Linear(d_model, d_qkv * n_head, bias=False)
        # self.w_q = nn.Linear(d_model, d_qkv * n_head, bias=False)
        # self.w_v = nn.Linear(d_model, d_qkv * n_head, bias=False)
        # self.w_out = nn.Linear(d_qkv * n_head, d_model)

        # The hyperparameters given as arguments to this function and others
        # should be sufficient to reach the score targets for this project

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """Runs the multi-head self-attention layer.

        Args:
          x: the input to the layer, a tensor of shape [batch size, length, d_model]
          mask: a mask for disallowing attention to padding tokens. You will need to
                construct the mask yourself further on in this notebook. You may
                implement masking in any way; there is no requirement that you use
                a particular form for the mask object.
        Returns:
          A single tensor containing the output from this layer
        """
        """YOUR CODE HERE"""
        # Implementation tip: using torch.einsum will greatly simplify the code that
        # you need to write.

        # Apply linear projections to convert the feature vector at each token into separate vectors for the query, key, and value.
        q = torch.einsum('blm,hmd->blhd', x, self.w_q)
        k = torch.einsum('blm,hmd->blhd', x, self.w_k)
        v = torch.einsum('blm,hmd->blhd', x, self.w_v)  # b l h d

        # Apply attention, scaling the logits by 1 / d_{kqv} .
        prod_key_queries = torch.einsum('blhd,bkhd->bhlk', q, k)

        # Ensure proper masking, such that padding tokens are never attended to.
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            prod_key_queries = prod_key_queries.masked_fill(
                mask == 0, float("-1e9"))

        # attention
        logits = torch.softmax(
            prod_key_queries/(self.d_model ** (1/2)), dim=3)  # b,h,l,k
        out = torch.einsum('bhlk,blhd->blhd', [logits, v])  # blhd
        # flattening last 2 dimensions
        out = out.reshape(x.size(0), x.size(1), self.n_head * self.d_qkv)
        out = self.fc_out(out)
        return out


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):

        residual = x.clone()
        z = self.w_1(x)
        y = self.relu(z)

        x = self.dropout(y)
        x = self.w_2(x)
        x += residual
        x = self.layer_norm(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=256, d_ff=1024, n_layers=4, n_head=4, d_qkv=32,
                 dropout=0.1):
        super().__init__()
        # Implementation tip: if you are storing nn.Module objects in a list, use
        # nn.ModuleList. If you use assignment statements of the form
        # `self.sublayers = [x, y, z]` with a plain python list instead of a
        # ModuleList, you might find that none of the sub-layer parameters are
        # trained.
        self.multiHeadAttention = MultiHeadAttention(
            d_model, n_head, d_qkv, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """Runs the Transformer encoder.

        Args:
          x: the input to the Transformer, a tensor of shape
             [batch size, length, d_model]
          mask: a mask for disallowing attention to padding tokens. You will need to
                construct the mask yourself further on in this notebook. You may
                implement masking in any way; there is no requirement that you use
                a particular form for the mask object.
        Returns:
          A single tensor containing the output from the Transformer
        """
        residual = x
        logits = self.multiHeadAttention(x, mask)
        forward = self.feed_forward(logits)
        out = self.dropout(forward)
        out += residual
        out = self.norm2(out)
        return out


class AddPositionalEncoding(nn.Module):
    def __init__(self, d_model=256, input_dropout=0.1, timing_dropout=0.1,
                 max_len=512):
        super().__init__()
        self.timing_table = nn.Parameter(torch.FloatTensor(max_len, d_model))
        nn.init.normal_(self.timing_table)
        self.input_dropout = nn.Dropout(input_dropout)
        self.timing_dropout = nn.Dropout(timing_dropout)

    def forward(self, x):
        """
        Args:
          x: A tensor of shape [batch size, length, d_model]
        """
        x = self.input_dropout(x)
        timing = self.timing_table[None, :x.shape[1], :]
        timing = self.timing_dropout(timing)
        return x + timing


class TransformerPOSTaggingModel(POSTaggingModel):
    def __init__(self, vocab, PARTS_OF_SPEECH):
        super().__init__()
        d_model = 256
        self.add_timing = AddPositionalEncoding(d_model)
        self.encoder = TransformerEncoder(d_model)
        self.stack = nn.ModuleList(
            [TransformerEncoder(d_model) for _ in range(2)])

        """more starting code."""
        self.PAD_ID = vocab.PieceToId("<pad>")
        self.embedding = nn.Embedding(vocab.GetPieceSize(), d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=0.1)

    def encode(self, batch):
        """
        Args:
          batch: an input batch as a dictionary; the key 'ids' holds the vocab ids
            of the subword tokens in a tensor of size [batch_size, sequence_length]
        Returns:
          A single tensor containing logits for each subword token
            You don't need to filter the unlabeled subwords - this is handled by our
            code above.
        """

        # Implementation tip: you will want to use another normalization layer
        # between the output of the encoder and the final projection layer

        """YOUR CODE HERE."""
        ids = batch['ids']
        mask = ids != self.PAD_ID

        x = self.embedding(ids)
        x = self.add_timing(x)
        #x = self.norm(self.dropout(x))

        for e in self.stack:
            x = e(x, mask)
        return x
