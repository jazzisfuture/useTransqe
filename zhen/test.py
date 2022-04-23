import os
import shutil
import sys
os.environ ["CUDA_VISIBLE_DEVICES"] = "1" 
sys.path.append('..')

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from qe.common.util.draw import draw_scatterplot, print_stat
from qe.common.util.normalizer import fit, un_fit
from qe.common.util.postprocess import format_submission
from qe.common.util.reader import read_annotated_file, read_test_file
from config import TEMP_DIRECTORY, MODEL_NAME, \
    monotransquest_config, MODEL_TYPE, SEED, RESULT_FILE, SUBMISSION_FILE, RESULT_IMAGE
from transquest.algo.sentence_level.monotransquest.evaluation import pearson_corr, spearman_corr
from transquest.algo.sentence_level.monotransquest.run_model import MonoTransQuestModel


if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

TRAIN_FOLDER = "/home/yljiang/code/qe/data/QE-sent/zh-en/train"
DEV_FOLDER = "/home/yljiang/code/qe/data/QE-sent/zh-en/dev"
TEST_FOLDER = "/home/yljiang/code/qe/data/test/CCMT2021_QE_CE_test"

train = read_annotated_file(path=TRAIN_FOLDER, original_file="train.src", translation_file="train.tgt",
                            hter_file="train.hter")
dev = read_annotated_file(path=DEV_FOLDER, original_file="new.dev.src", translation_file="new.dev.tgt", hter_file="new.dev.hter")
test = read_test_file(path=TEST_FOLDER, original_file="source.txt", translation_file="target.txt")

train = train[['original', 'translation', 'hter']]
dev = dev[['original', 'translation', 'hter']]
test = test[['index', 'original', 'translation']]

index = test['index'].to_list()
train = train.rename(columns={'original': 'text_a', 'translation': 'text_b', 'hter': 'labels'}).dropna()
dev = dev.rename(columns={'original': 'text_a', 'translation': 'text_b', 'hter': 'labels'}).dropna()
test = test.rename(columns={'original': 'text_a', 'translation': 'text_b'}).dropna()
best = "/home/yljiang/code/qe/temp/outputs/best_model"

model = MonoTransQuestModel(MODEL_TYPE, best, num_labels=1, use_cuda=torch.cuda.is_available())

result_test, model_outputs_test, wrong_predictions_test = model.eval_model(test, pearson_corr=pearson_corr,
                                                                    spearman_corr=spearman_corr,
                                                                    mae=mean_absolute_error)

print(result_test,model_outputs_test,wrong_predictions_test)