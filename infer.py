# -*- coding: utf-8 -*-

import os
import sys
import operator
import numpy as np
import torch
import argparse
from loguru import logger

sys.path.append('../..')

from mutiCnnS2S import MutiCnnS2S

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unk_tokens = [' ', '“', '”', '‘', '’', '琊', '\n', '…', '—', '擤', '\t', '֍', '玕', '', '《', '》']
# Define constants associated with the usual special tokens.
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'

def load_word_dict(save_path):
    dict_data = dict()
    num = 0
    with open(save_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            items = line.split('\t')
            num += 1
            try:
                dict_data[items[0]] = int(items[1])
            except IndexError:
                logger.error('IndexError, index:%s, line:%s' % (num, line))
    return dict_data

def get_errors(corrected_text, origin_text):
    sub_details = []
    for i, ori_char in enumerate(origin_text):
        if i >= len(corrected_text):
            continue
        if ori_char in unk_tokens:
            # deal with unk word
            corrected_text = corrected_text[:i] + ori_char + corrected_text[i:]
            continue
        if ori_char != corrected_text[i]:
            sub_details.append((ori_char, corrected_text[i], i, i + 1))
    sub_details = sorted(sub_details, key=operator.itemgetter(2))
    return corrected_text, sub_details


class Inference(object):
    def __init__(self, model_dir, arch='convseq2seq',
                 embed_size=128, hidden_size=128, dropout=0.25, max_length=128):
        logger.debug("Device: {}".format(device))
        logger.debug(f'Use {arch} model.')
        if arch =='convseq2seq':
            src_vocab_path = os.path.join(model_dir, 'vocab_source.txt')
            trg_vocab_path = os.path.join(model_dir, 'vocab_target.txt')
            self.src_2_ids = load_word_dict(src_vocab_path)
            self.trg_2_ids = load_word_dict(trg_vocab_path)
            self.id_2_trgs = {v: k for k, v in self.trg_2_ids.items()}
            trg_pad_idx = self.trg_2_ids[PAD_TOKEN]
            self.model = MutiCnnS2S(encoder_vocab_size=len(self.src_2_ids),
                                        decoder_vocab_size=len(self.trg_2_ids),
                                        embed_size=embed_size,
                                        enc_hidden_size=hidden_size,
                                        dec_hidden_size=hidden_size,
                                        dropout=dropout,
                                        trg_pad_idx=trg_pad_idx,
                                        device=device,
                                        max_length=max_length).to(device)
            model_path = os.path.join(model_dir, 'convseq2seq.pth')
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            logger.debug('Load model from {}'.format(model_path))
            self.model.eval()
        else:
            logger.error('error arch: {}'.format(arch))
            raise ValueError("Model arch choose error. Must use one of seq2seq model.")
        self.arch = arch
        self.max_length = max_length

    def predict(self, sentence_list):
        result = []
        if self.arch =='convseq2seq':
            for query in sentence_list:
                out = []
                tokens = [token.lower() for token in query]
                tokens = [SOS_TOKEN] + tokens + [EOS_TOKEN]
                src_ids = [self.src_2_ids[i] for i in tokens if i in self.src_2_ids]

                sos_idx = self.trg_2_ids[SOS_TOKEN]
                src_tensor = torch.from_numpy(np.array(src_ids).reshape(1, -1)).long().to(device)
                translation, attn = self.model.translate(src_tensor, sos_idx)
                translation = [self.id_2_trgs[i] for i in translation if i in self.id_2_trgs]
                for word in translation:
                    if word != EOS_TOKEN:
                        out.append(word)
                    else:
                        break
                corrected_text = ''.join(out)
                corrected_text, sub_details = get_errors(corrected_text, query)
                result.append([corrected_text, sub_details])
        return result


def Corrector(inputs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="output/hyt_convSeq2seq/", type=str, help="Dir for model save.")
    parser.add_argument("--arch", default="convseq2seq", type=str,
                        help="The name of the task to train selected in the list: " + ", ".join(
                            ['seq2seq', 'convseq2seq', 'bertseq2seq']),
                        )
    parser.add_argument("--max_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. \n"
                             "Sequences longer than this will be truncated, sequences shorter padded.",
                        )
    parser.add_argument("--embed_size", default=128, type=int, help="Embedding size.")
    parser.add_argument("--hidden_size", default=128, type=int, help="Hidden size.")
    parser.add_argument("--dropout", default=0.25, type=float, help="Dropout rate.")

    args = parser.parse_args()
    print(args)

    m = Inference(args.model_dir,
                  args.arch,
                  embed_size=args.embed_size,
                  hidden_size=args.hidden_size,
                  dropout=args.dropout,
                  max_length=args.max_length
                  )

    outputs = m.predict(inputs)
    return outputs

