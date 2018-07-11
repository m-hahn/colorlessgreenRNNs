# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import logging
import math
import time

import torch
import torch.nn as nn

from dictionary_corpus import Corpus
import model
from lm_argparser import lm_parser
from utils import repackage_hidden, get_batch, batchify

parser = argparse.ArgumentParser(parents=[lm_parser],
                                 description="Basic training and evaluation for RNN LM")

args = parser.parse_args()

logging.basicConfig(level=logging.INFO, handlers=[#logging.StreamHandler(),
                                                  logging.FileHandler(args.log)])
logging.info(args)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

logging.info("Loading data")
start = time.time()
corpus = Corpus(args.data, onlyTest=True)
logging.info("( %.2f )" % (time.time() - start))
#logging.info(corpus.train)

logging.info("Batchying..")
eval_batch_size = 1
#train_data = batchify(corpus.train, args.batch_size, args.cuda)
#logging.info("Train data size", train_data.size())
#val_data = batchify(corpus.valid, eval_batch_size, args.cuda)
test_data = batchify(corpus.test, eval_batch_size, args.cuda)

ntokens = len(corpus.dictionary)

criterion = nn.CrossEntropyLoss()

###############################################################################
# Build the model
###############################################################################

logging.info("Building the model")

model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
if args.cuda:
    model.cuda()

with open(args.save, 'rb') as f:
    model = torch.load(f)



###############################################################################
# Training code
###############################################################################


def compute_surprisals(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(eval_batch_size)

    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args.bptt, evaluation=True)
#        print(data.size())
 #       print(targets.size())
        #> output has size seq_length x batch_size x vocab_size
        output, hidden = model(data, None)
        #> output_flat has size num_targets x vocab_size (batches are stacked together)
        #> ! important, otherwise softmax computation (e.g. with F.softmax()) is incorrect
        output_flat = output.view(-1, ntokens)
        #output_candidates_info(output_flat.data, targets.data)
        loss = nn.CrossEntropyLoss(size_average=False, reduce=False)(output_flat, targets).data.cpu().view(-1, eval_batch_size).numpy()
#        print(loss)
#        hidden = repackage_hidden(hidden)
        targets = targets.data.view(-1, eval_batch_size).cpu().numpy()
        for batch in range(eval_batch_size):
           print("--")
           for j in range(data.size()[0]):
             print(corpus.dictionary.idx2word[targets[j][batch]], loss[j][batch])
        print(loss.mean())
    return total_loss[0] /len(data_source)


compute_surprisals(test_data)




