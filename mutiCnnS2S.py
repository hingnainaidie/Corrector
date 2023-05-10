# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# def get_attn_pad_mask(seq_q, seq_k):
#     '''
#     seq_q: [batch_size, seq_len]
#     seq_k: [batch_size, seq_len]
#     seq_len could be src_len or it could be tgt_len
#     seq_len in seq_q and seq_len in seq_k maybe not equal
#     '''
#     batch_size, len_q = seq_q.size()
#     batch_size, len_k = seq_k.size()
#     # eq(zero) is PAD token
#     pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
#     return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(64) # scores : [batch_size, n_heads, len_q, len_k]
        # scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn

class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 emb_dim=256,
                 hid_dim=512,
                 n_layers=2,
                 kernel_size=3,
                 dropout=0.25,
                 device=torch.device('cuda'),
                 max_length=128):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd!"
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.tok_embedding = nn.Embedding(input_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)
        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=hid_dim,
                                              out_channels=2 * hid_dim,
                                              kernel_size=kernel_size,
                                              padding=(kernel_size - 1) // 2)
                                    for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [batch size, src len]
        batch_size = src.shape[0]
        src_len = src.shape[1]
        # create position tensor
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos = [0, 1, 2, 3, ..., src len - 1]
        # pos = [batch size, src len]
        # embed tokens and positions
        tok_embedded = self.tok_embedding(src)
        pos_embedded = self.pos_embedding(pos)
        # tok_embedded = pos_embedded = [batch size, src len, emb dim]
        # combine embeddings by elementwise summing
        embedded = self.dropout(tok_embedded + pos_embedded)
        # embedded = [batch size, src len, emb dim]
        # pass embedded through linear layer to convert from emb dim to hid dim
        conv_input = self.emb2hid(embedded)
        # conv_input = [batch size, src len, hid dim]
        # permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)
        # conv_input = [batch size, hid dim, src len]
        # begin convolutional blocks...
        for i, conv in enumerate(self.convs):
            # pass through convolutional layer
            conved = conv(self.dropout(conv_input))
            # conved = [batch size, 2 * hid dim, src len]
            # pass through GLU activation function
            conved = F.glu(conved, dim=1)
            # conved = [batch size, hid dim, src len]
            # apply residual connection
            conved = (conved + conv_input) * self.scale
            # conved = [batch size, hid dim, src len]
            # set conv_input to conved for next loop iteration
            conv_input = conved
        # end convolutional blocks
        # permute and convert back to emb dim
        conved = self.hid2emb(conved.permute(0, 2, 1))
        # conved = [batch size, src len, emb dim]
        # elementwise sum output (conved) and input (embedded) to be used for attention
        combined = (conved + embedded) * self.scale
        # combined = [batch size, src len, emb dim]
        return conved, combined


class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 emb_dim=256,
                 hid_dim=512,
                 n_layers=2,
                 kernel_size=3,
                 dropout=0.25,
                 trg_pad_idx=0,
                 device=torch.device('cuda'),
                 max_length=128):
        super().__init__()
        self.hid_dim=hid_dim
        self.kernel_size = kernel_size
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.tok_embedding = nn.Embedding(output_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)
        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)
        self.attn_hid2emb = nn.Linear(hid_dim, emb_dim)

        self.W_Q=nn.Linear(emb_dim,8*64,bias=False)
        self.W_K=nn.Linear(emb_dim,8*64,bias=False)
        self.W_V=nn.Linear(emb_dim,8*64,bias=False)
        self.headSum = nn.Linear(8*64,hid_dim, bias=False)

        self.attn_emb2hid = nn.Linear(emb_dim, hid_dim)
        self.fc_out = nn.Linear(emb_dim, output_dim)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=hid_dim,
                                              out_channels=2 * hid_dim,
                                              kernel_size=kernel_size)
                                    for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    # def calculate_attention(self, embedded, conved, encoder_conved, encoder_combined):
    #     """
    #     Attention
    #     :param embedded: embedded = [batch size, trg len, emb dim]
    #     :param conved: conved = [batch size, hid dim, trg len]
    #     :param encoder_conved: encoder_conved = encoder_combined = [batch size, src len, emb dim]
    #     :param encoder_combined: permute and convert back to emb dim
    #     :return:
    #     """
    #     conved_emb = self.attn_hid2emb(conved.permute(0, 2, 1))
    #     # conved_emb = [batch size, trg len, emb dim]
    #     combined = (conved_emb + embedded) * self.scale
    #     # combined = [batch size, trg len, emb dim]
    #     energy = torch.matmul(combined, encoder_conved.permute(0, 2, 1))
    #     # energy = [batch size, trg len, src len]
    #     attention = F.softmax(energy, dim=2)
    #     # attention = [batch size, trg len, src len]
    #     attended_encoding = torch.matmul(attention, encoder_combined)
    #     # attended_encoding = [batch size, trg len, emd dim]
    #     # convert from emb dim -> hid dim
    #     attended_encoding = self.attn_emb2hid(attended_encoding)
    #     # attended_encoding = [batch size, trg len, hid dim]
    #     # apply residual connection
    #     attended_combined = (conved + attended_encoding.permute(0, 2, 1)) * self.scale
    #     # attended_combined = [batch size, hid dim, trg len]
    #     return attention, attended_combined

    def calculate_attention(self, embedded, conved, encoder_conved, encoder_combined):
        """
        Attention
        :param embedded: embedded = [batch size, trg len, emb dim]
        :param conved: conved = [batch size, hid dim, trg len]
        :param encoder_conved: encoder_conved = encoder_combined = [batch size, src len, emb dim]
        :param encoder_combined: permute and convert back to emb dim
        :return:
        """
        conved_emb = self.attn_hid2emb(conved.permute(0, 2, 1))
        # conved_emb = [batch size, trg len, emb dim]
        combined = (conved_emb + embedded) * self.scale #将卷积得到的矩阵数据和原数据相加
        # combined = [batch size, trg len, emb dim]
        batch_size=combined.size(0)
        Q=self.W_Q(combined).view(batch_size,-1,8,64).transpose(1,2)   #8为注意力多头数量，64是d_k,d_q,d_v的维度
        # Q=[batch_size,n_head,trg_len,d_q]
        K=self.W_K(encoder_conved).view(batch_size,-1,8,64).transpose(1,2)
        # K=[batch_size,n_head,src_len,d_q]
        V=self.W_V(encoder_conved).view(batch_size,-1,8,64).transpose(1,2)
        # V=[batch_size,n_head,src_len,d_q]
        
        # attn_mask=get_attn_pad_mask(trg,src).unsqueeze(1).repeat(1, 8, 1, 1)
        context, attn = ScaledDotProductAttention()(Q, K, V)
        #context[batch_size, n_heads, trg_len, d_v]
        #attn[batch_size, n_heads, trg_len, d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, 8 * 64)
        # context: [batch_size, trg_len, n_heads * d_v]
        output = self.headSum(context).transpose(1,2)
        #[batch size, hid dim, trg len]
        # return attn,nn.LayerNorm(self.hid_dim)(output + conved)
        return attn,(output + conved)*self.scale

    def forward(self, trg, encoder_conved, encoder_combined):
        """
        Get output and attention
        :param trg: trg = [batch size, trg len]
        :param encoder_conved: encoder_conved = encoder_combined = [batch size, src len, emb dim]
        :param encoder_combined:
        :return:
        """
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        # create position tensor
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos = [batch size, trg len]
        # embed tokens and positions
        tok_embedded = self.tok_embedding(trg)
        pos_embedded = self.pos_embedding(pos)
        # tok_embedded = [batch size, trg len, emb dim]
        # pos_embedded = [batch size, trg len, emb dim]
        # combine embeddings by elementwise summing
        embedded = self.dropout(tok_embedded + pos_embedded)
        # embedded = [batch size, trg len, emb dim]
        # pass embedded through linear layer to go through emb dim -> hid dim
        conv_input = self.emb2hid(embedded)
        # conv_input = [batch size, trg len, hid dim]
        # permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)
        # conv_input = [batch size, hid dim, trg len]
        batch_size = conv_input.shape[0]
        hid_dim = conv_input.shape[1]
        for i, conv in enumerate(self.convs):
            # apply dropout
            conv_input = self.dropout(conv_input)
            # need to pad so decoder can't "cheat"
            padding = torch.zeros(batch_size,
                                  hid_dim,
                                  self.kernel_size - 1).fill_(self.trg_pad_idx).to(self.device)
            padded_conv_input = torch.cat((padding, conv_input), dim=2)
            # padded_conv_input = [batch size, hid dim, trg len + kernel size - 1]
            # pass through convolutional layer
            conved = conv(padded_conv_input)
            # conved = [batch size, 2 * hid dim, trg len]
            # pass through GLU activation function
            conved = F.glu(conved, dim=1)
            # conved = [batch size, hid dim, trg len]
            # calculate attention
            attention, conved = self.calculate_attention(embedded,
                                                         conved,
                                                         encoder_conved,
                                                         encoder_combined)
            # attention = [batch_size, n_heads, trg_len, d_v]
            # apply residual connection
            conved = (conved + conv_input) * self.scale
            # conved = [batch size, hid dim, trg len]
            # set conv_input to conved for next loop iteration
            conv_input = conved
        conved = self.hid2emb(conved.permute(0, 2, 1))
        # conved = [batch size, trg len, emb dim]
        output = self.fc_out(self.dropout(conved))
        # output = [batch size, trg len, output dim]
        return output, attention


class MutiCnnS2S(nn.Module):
    def __init__(self,
                 encoder_vocab_size,
                 decoder_vocab_size,
                 embed_size,
                 enc_hidden_size,
                 dec_hidden_size,
                 dropout,
                 trg_pad_idx,
                 device,
                 max_length=128
                 ):
        super().__init__()
        self.encoder = Encoder(input_dim=encoder_vocab_size,
                               emb_dim=embed_size,
                               hid_dim=enc_hidden_size,
                               n_layers=2,
                               kernel_size=3,
                               dropout=dropout,
                               device=device,
                               max_length=max_length)
        self.decoder = Decoder(output_dim=decoder_vocab_size,
                               emb_dim=embed_size,
                               hid_dim=dec_hidden_size,
                               n_layers=2,
                               kernel_size=3,
                               dropout=dropout,
                               trg_pad_idx=trg_pad_idx,
                               device=device,
                               max_length=max_length)
        self.max_length = max_length
        self.device = device

    def forward(self, src, trg):
        """
        Calculate z^u (encoder_conved) and (z^u + e) (encoder_combined)
        :param src:src = [batch size, src len]
        :param trg: trg = [batch size, trg len - 1] (<eos> token sliced off the end)
        :return:
        """
        # encoder_conved is output from final encoder conv. block
        # encoder_combined is encoder_conved plus (elementwise) src embedding plus
        #  positional embeddings
        encoder_conved, encoder_combined = self.encoder(src)
        # encoder_conved = [batch size, src len, emb dim]
        # encoder_combined = [batch size, src len, emb dim]
        # calculate predictions of next words
        # output is a batch of predictions for each word in the trg sentence
        # attention a batch of attention scores across the src sentence for
        #  each word in the trg sentence
        output, attention = self.decoder(trg, encoder_conved, encoder_combined)
        # output = [batch size, trg len - 1, output dim]
        # attention = [batch size, trg len - 1, src len]
        return output, attention

    def translate(self, x, sos):
        """
        Predict x
        :param x: input tensor
        :param sos: SOS tensor
        :return: preds, attns
        """
        encoder_conved, encoder_combined = self.encoder(x)
        preds = []
        attns = []
        trg_indexes = [sos]
        for i in range(self.max_length):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(self.device)
            output, attention = self.decoder(trg_tensor, encoder_conved, encoder_combined)
            pred = output.argmax(2)[:, -1].item()
            preds.append(pred)
            attns.append(attention)
            trg_indexes.append(pred)

        return preds, attns