# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import json
import argparse
from collections import OrderedDict
import torch

from src.utils import bool_flag, initialize_exp
from src.models import build_model
from src.trainer import Trainer
from src.evaluation import Evaluator
import wandb


VALIDATION_METRIC_SUP = 'precision_at_1-csls_knn_100'
VALIDATION_METRIC_UNSUP = 'mean_cosine-csls_knn_100-S2T-10000'

PARAMETERS_NUM = {
    'resnet18':11.69, 'resnet34':21.80, 'resnet50':25.56, 'resnet101':44.55, 'resnet152':60.19,
    'bert_uncased_L-2_H-128_A-2':4.4, 'bert_uncased_L-4_H-256_A-4':11.3, 'bert_uncased_L-4_H-512_A-8':29.1, 
    'bert_uncased_L-8_H-512_A-8':41.7, 'bert-base-uncased':110, 'bert-large-uncased':340,
    'gpt2':117, 'gpt2-medium':345, 'gpt2-large':762, 'gpt2-xl':1542,
    'opt-125m':125, 'opt-1.3b':1300, 'opt-6.7b':6700, 'opt-30b':30000,
    "segformer-b0-finetuned-ade-512-512":3.4,
    "segformer-b1-finetuned-ade-512-512":13.1,
    "segformer-b2-finetuned-ade-512-512":24.2,
    "segformer-b3-finetuned-ade-512-512":44.0,
    "segformer-b4-finetuned-ade-512-512":60.8,
    "segformer-b5-finetuned-ade-640-640":81.4}

# main
parser = argparse.ArgumentParser(description='Supervised training')
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
parser.add_argument("--export", type=str, default="", help="Export embeddings after training (txt / pth)")

# data
parser.add_argument("--src_lang", type=str, default='en', help="Source language")
parser.add_argument("--tgt_lang", type=str, default='es', help="Target language")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
# training refinement
parser.add_argument("--n_refinement", type=int, default=5, help="Number of refinement iterations (0 to disable the refinement procedure)")
# dictionary creation parameters (for refinement)
parser.add_argument("--dico_train", type=str, default="default", help="Path to training dictionary (default: use identical character strings)")
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
parser.add_argument("--dico_method", type=str, default='csls_knn_100', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
parser.add_argument("--dico_build", type=str, default='S2T&T2S', help="S2T,T2S,S2T|T2S,S2T&T2S")
parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
parser.add_argument("--dico_max_rank", type=int, default=10000, help="Maximum dictionary words rank (0 to disable)")
parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
# reload pre-trained embeddings
parser.add_argument("--src_emb", type=str, default='', help="Reload source embeddings")
parser.add_argument("--tgt_emb", type=str, default='', help="Reload target embeddings")
parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")
parser.add_argument("--load_optim", type=bool_flag, default=False, help="Reload optimal")

# parse parameters
params = parser.parse_args()

# check parameters
assert not params.cuda or torch.cuda.is_available()
assert params.dico_train in ["identical_char", "default"] or os.path.isfile(params.dico_train)
assert params.dico_build in ["S2T", "T2S", "S2T|T2S", "S2T&T2S"]
assert params.dico_max_size == 0 or params.dico_max_size < params.dico_max_rank
assert params.dico_max_size == 0 or params.dico_max_size > params.dico_min_size
assert os.path.isfile(params.src_emb)
assert os.path.isfile(params.tgt_emb)
assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)
assert params.export in ["", "txt", "pth"]

if 'phrase' in params.dico_eval :
    polysemy_type = 'PHRASE'
elif '2to3' in params.dico_eval:
    polysemy_type = '2&3'
elif 'unk' in params.dico_eval:
    polysemy_type = 'UNKNOWN'
elif 'single' in params.dico_eval:
    polysemy_type = '1'
else:
    polysemy_type = '4+'

wandb.init(
    project="muse_supervised_id2words", 
    name=f"seed_{params.seed}_{params.src_lang}_{params.tgt_lang}",
    group=f"{params.src_lang}_{params.tgt_lang}",
    tags=[f"{params.src_lang}",f"{params.tgt_lang}"]
    )

# build logger / model / trainer / evaluator
logger = initialize_exp(params)
src_emb, tgt_emb, mapping, _ = build_model(params, False)
trainer = Trainer(src_emb, tgt_emb, mapping, None, params)
evaluator = Evaluator(trainer)

tgt_parameter_size = [PARAMETERS_NUM[f'{params.tgt_lang}']]
src_parameter_size = [PARAMETERS_NUM[f'{params.src_lang}']]
# print(parameter_size)
# print(parameter_size * 3)
# load a training dictionary. if a dictionary path is not provided, use a default
# one ("default") or create one based on identical character strings ("identical_char")
trainer.load_training_dico(params.dico_train)

# define the validation metric
VALIDATION_METRIC = VALIDATION_METRIC_UNSUP if params.dico_train == 'identical_char' else VALIDATION_METRIC_SUP
logger.info("Validation metric: %s" % VALIDATION_METRIC)

"""
Learning loop for Procrustes Iterative Learning
"""
for n_iter in range(params.n_refinement + 1):

    logger.info('Starting iteration %i...' % n_iter)

    # build a dictionary from aligned embeddings (unless
    # it is the first iteration and we use the init one)
    if n_iter > 0 or not hasattr(trainer, 'dico'):
        trainer.build_dictionary()

    # apply the Procrustes solution
    trainer.procrustes()

    # embeddings evaluation
    to_log = OrderedDict({'n_iter': n_iter})
    evaluator.all_eval(to_log)
    wandb.log({
        # "precision_at_1-nn": to_log["precision_at_1-nn"], 
        # "precision_at_10-nn":to_log["precision_at_10-nn"] , 
        # "precision_at_100-nn":to_log["precision_at_100-nn"] , 
        "precision_at_1-csls_knn_100":to_log["precision_at_1-csls_knn_100"] , 
        "precision_at_10-csls_knn_100":to_log["precision_at_10-csls_knn_100"] , 
        "precision_at_100-csls_knn_100":to_log["precision_at_100-csls_knn_100"] , 
        # "Meanings":polysemy_type,
        # "tgt_parameters_num":tgt_parameter_size[0],
        # "src_parameters_num":src_parameter_size[0],
    })

    # JSON log / save best model / end of epoch
    logger.info("__log__:%s" % json.dumps(to_log))
    # trainer.save_best(to_log, VALIDATION_METRIC)
    logger.info('End of iteration %i.\n\n' % n_iter)


# export embeddings
if params.export:
    trainer.reload_best()
    trainer.export()
