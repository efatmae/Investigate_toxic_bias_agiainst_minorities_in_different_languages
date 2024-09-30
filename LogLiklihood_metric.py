import os
import csv
import json
import math
import torch
import argparse
import difflib
import logging
import numpy as np
import pandas as pd
from numpy.matlib import identity
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import BertTokenizer, BertForMaskedLM
from transformers import AlbertTokenizer, AlbertForMaskedLM
from transformers import RobertaTokenizer, RobertaForMaskedLM
from collections import defaultdict
from tqdm import tqdm


def read_data(input_file):
    """
    Load data into pandas DataFrame format.
    """
    
    df_data = pd.DataFrame(columns=['sent1', 'sent2', 'direction', 'bias_type', 'identity'])

    with open(input_file) as f:
        reader = csv.DictReader(f)
        i = 0
        print("read input file with csv package")
        for row in reader:
            direction, gold_bias = '_', '_'
            direction = row['stereo_antistereo']
            bias_type = row['bias_type']
            identity = row["identity"]
            sent1, sent2 = '', ''
            if direction == 'stereo':
                sent1 = row['Toxic_sent']
                sent2 = row['Non_Toxic_sent']
            else:
                sent1 = row['Non_Toxic_sent']
                sent2 = row['Toxic_sent']
            #print("sent1", sent1)
            #print("sent2", sent2)
            df_item = pd.DataFrame({'sent1': sent1,
                       'sent2': sent2,
                       'direction': direction,
                       'bias_type': bias_type,
                       'identity':  identity}, index=[i])
            df_data = pd.concat([df_data, df_item], ignore_index=True)
            i+=1
        print(i)
    return df_data


def get_log_prob_unigram(masked_token_ids, token_ids, mask_idx, lm):
    """
    Given a sequence of token ids, with one masked token, return the log probability of the masked token.
    """
    
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    log_softmax = lm["log_softmax"]
    mask_token = lm["mask_token"]
    uncased = lm["uncased"]

    # get model hidden states
    output = model(masked_token_ids)
    hidden_states = output[0].squeeze(0)
    print("hidden_states.shape", hidden_states.shape)
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)

    # we only need log_prob for the MASK tokens
    print("mask_idx", mask_idx)
    assert masked_token_ids[0][mask_idx] == mask_id

    hs = hidden_states[mask_idx]
    print("hs.shape", hs.shape)
    print("token_ids len", len(token_ids[0]))
    print("token_ids [0][1] shape", token_ids)
    target_id = token_ids[0][1]
    print("target_id",target_id)
    log_probs = log_softmax(hs)[target_id]

    return log_probs


def get_span(seq1, seq2):
    """
    This function extract spans that are shared between two sequences.
    """

    seq1 = [str(x) for x in seq1.tolist()]
    seq2 = [str(x) for x in seq2.tolist()]

    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    template1, template2 = [], []
    for op in matcher.get_opcodes():
        # each op is a list of tuple: 
        # (operation, pro_idx_start, pro_idx_end, anti_idx_start, anti_idx_end)
        # possible operation: replace, insert, equal
        # https://docs.python.org/3/library/difflib.html
        if op[0] == 'equal':
            template1 += [x for x in range(op[1], op[2], 1)]
            template2 += [x for x in range(op[3], op[4], 1)]

    return template1, template2


def mask_unigram(data, lm, n=1):
    """
    Score each sentence by masking one word at a time.
    The score for a sentence is the sum of log probability of each word in
    the sentence.
    n = n-gram of token that is masked, if n > 1, we mask tokens with overlapping
    n-grams.
    """
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    log_softmax = lm["log_softmax"]
    mask_token = lm["mask_token"]
    uncased = lm["uncased"]

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    sent1, sent2 = data["sent1"], data["sent2"]
    print("sent1", sent1)
    print("sent2", sent2)
    if uncased:
        sent1 = sent1.lower()
        sent2 = sent2.lower()

    # tokenize
    sent1_token_ids = tokenizer.encode(sent1, return_tensors='pt')
    sent2_token_ids = tokenizer.encode(sent2, return_tensors='pt')

    print("sent1_token_ids", sent1_token_ids)
    print("sent2_token_ids", sent2_token_ids)

    #output1 = model(**sent1_token_ids)
    #print("output1", output1.last_hidden_state.shape)

    #output2 = model(**sent2_token_ids)
    #print("output2", output2.last_hidden_state.shape)
    # get spans of non-changing tokens
    template1, template2 = get_span(sent1_token_ids[0], sent2_token_ids[0])

    assert len(template1) == len(template2)

    N = len(template1)  # num. of tokens that can be masked
    print("No. tokens", N)
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)
    print("mask_id", mask_id)
    sent1_log_probs = 0.
    sent2_log_probs = 0.
    total_masked_tokens = 0

    # skipping CLS and SEP tokens, they'll never be masked
    for i in range(1, N-1):
        sent1_masked_token_ids = sent1_token_ids.clone().detach()
        print("sent1_masked_token_ids", len(sent1_masked_token_ids))
        sent2_masked_token_ids = sent2_token_ids.clone().detach()
        print("sent2_masked_token_ids", len(sent2_masked_token_ids))

        sent1_masked_token_ids[0][template1[i]] = mask_id
        sent2_masked_token_ids[0][template2[i]] = mask_id
        total_masked_tokens += 1

        print("template1[i]", template1[i])
        score1 = get_log_prob_unigram(sent1_masked_token_ids, sent1_token_ids, template1[i], lm)
        score2 = get_log_prob_unigram(sent2_masked_token_ids, sent2_token_ids, template2[i], lm)

        sent1_log_probs += score1.item()
        sent2_log_probs += score2.item()

    score = {}
    # average over iterations
    score["sent1_score"] = sent1_log_probs
    score["sent2_score"] = sent2_log_probs

    return score


def evaluate(args):
    """
    Evaluate a masked language model using CrowS-Pairs dataset.
    """

    print("Evaluating:")
    print("Input:", args.input_file)
    print("Model:", args.lm_model)
    print("Model_path:", args.lm_model_path)
    model_path = args.lm_model_path
    print("=" * 100)

    logging.basicConfig(level=logging.INFO)

    # load data into panda DataFrame
    df_data = read_data(args.input_file)

    # supported masked language models
    if args.lm_model == "bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        uncased = True
    if args.lm_model == "bert-large":
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        model = BertForMaskedLM.from_pretrained('bert-large-uncased')
        uncased = True
    elif args.lm_model == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        model = RobertaForMaskedLM.from_pretrained('roberta-large')
        uncased = False
    elif args.lm_model == "roberta-base":
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForMaskedLM.from_pretrained('roberta-base')
        uncased = False
    elif args.lm_model == "albert":
        tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v2')
        model = AlbertForMaskedLM.from_pretrained('albert-xxlarge-v2')
        uncased = True
    elif args.lm_model == "albert-base-v2":
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        model = AlbertForMaskedLM.from_pretrained('albert-base-v2')
        uncased = True
    elif args.lm_model == "other":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForMaskedLM.from_pretrained(model_path)
        uncased = False

    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')

    mask_token = tokenizer.mask_token
    log_softmax = torch.nn.LogSoftmax(dim=0)
    vocab = tokenizer.get_vocab()
    with open(args.lm_model + ".vocab", "w") as f:
        f.write(json.dumps(vocab))

    lm = {"model": model,
          "tokenizer": tokenizer,
          "mask_token": mask_token,
          "log_softmax": log_softmax,
          "uncased": uncased
    }

    # score each sentence. 
    # each row in the dataframe has the sentid and score for pro and anti stereo.
    df_score = pd.DataFrame(columns=['Toxic_sent', 'Non_Toxic_sent',
                                     'Toxic_sent_score', 'Non_Toxic_sent_score',
                                     'score', 'stereo_antistereo', 'bias_type', 'identity'])


    total_stereo, total_antistereo = 0, 0
    stereo_score, antistereo_score = 0, 0

    N = 0
    neutral = 0
    total = len(df_data.index)
    with tqdm(total=total) as pbar:
        j=0
        for index, data in df_data.iterrows():
            direction = data['direction']
            bias = data['bias_type']
            identity = data['identity']
            score = mask_unigram(data, lm)

            for stype in score.keys():
                score[stype] = round(score[stype], 3)

            N += 1
            pair_score = 0
            pbar.update(1)
            if score['sent1_score'] == score['sent2_score']:
                neutral += 1
            else:
                if direction == 'stereo':
                    total_stereo += 1
                    if score['sent1_score'] > score['sent2_score']:
                        stereo_score += 1
                        pair_score = 1
                elif direction == 'antistereo':
                    total_antistereo += 1
                    if score['sent2_score'] > score['sent1_score']:
                        antistereo_score += 1
                        pair_score = 1

            sent_more, sent_less = '', ''
            if direction == 'stereo':
                sent_more = data['sent1']
                sent_less = data['sent2']
                sent_more_score = score['sent1_score']
                sent_less_score = score['sent2_score']
            else:
                sent_more = data['sent2']
                sent_less = data['sent1']
                sent_more_score = score['sent2_score']
                sent_less_score = score['sent1_score']
            row_data = pd.DataFrame({'Toxic_sent': sent_more,
                                       'Non_Toxic_sent': sent_less,
                                       'Toxic_sent_score': sent_more_score,
                                       'Non_Toxic_sent_score': sent_less_score,
                                       'score': pair_score,
                                       'stereo_antistereo': direction,
                                       'bias_type': bias,
                                       'identity': identity}, index = [j])
            df_score = pd.concat([df_score,row_data], ignore_index=True)
            j+=1
    print(j)
    df_score.to_csv(args.output_file)
    print('=' * 100)
    print('Total examples:', N)
    print('Metric score:', round((stereo_score + antistereo_score) / N * 100, 2))
    print('Stereotype score:', round(stereo_score  / total_stereo * 100, 2))
    if antistereo_score != 0:
        print('Anti-stereotype score:', round(antistereo_score  / total_antistereo * 100, 2))
    print("Num. neutral:", neutral, round(neutral / N * 100, 2))
    print('=' * 100)
    print()


parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, help="path to input file")
parser.add_argument("--lm_model", type=str, help="pretrained LM model to use (options: bert, roberta, albert, other)")
parser.add_argument("--lm_model_path", type=str, help="path to the model if not available on HuggingFace", required=False)
parser.add_argument("--output_file", type=str, help="path to output file with sentence scores")

args = parser.parse_args()
evaluate(args)
