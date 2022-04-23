import os
import shutil
import sys
# os.environ ["CUDA_VISIBLE_DEVICES"] = "5" 
sys.path.append('..')

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from common.util.draw import draw_scatterplot, print_stat
from common.util.normalizer import fit, un_fit
from common.util.postprocess import format_submission
from common.util.reader import read_annotated_file, read_test_file
from config import TEMP_DIRECTORY, MODEL_NAME, \
    monotransquest_config, MODEL_TYPE, SEED, RESULT_FILE, SUBMISSION_FILE, RESULT_IMAGE
from transquest.algo.sentence_level.monotransquest.evaluation import pearson_corr, spearman_corr
from transquest.algo.sentence_level.monotransquest.run_model import MonoTransQuestModel


if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

TRAIN_FOLDER = "/root/qe/data/QE-sent/zh-en/train"
DEV_FOLDER = "/root/qe/data/QE-sent/zh-en/dev"
TEST_FOLDER = "/root/qe/data/test/CCMT2021_QE_CE_test"

train = read_annotated_file(path=TRAIN_FOLDER, original_file="train.src.original", translation_file="train.tgt.original",
                            hter_file="train.hter")
dev = read_annotated_file(path=DEV_FOLDER, original_file="dev.src.original", translation_file="dev.tgt.original", hter_file="dev.hter")
dev_test = read_annotated_file(path=DEV_FOLDER, original_file="test.src.original", translation_file="test.tgt.original", hter_file="test.hter")
test = read_test_file(path=TEST_FOLDER, original_file="source.txt", translation_file="target.txt")

train = train[['original', 'translation', 'hter']]
dev = dev[['original', 'translation', 'hter']]
dev_test= dev_test[['original', 'translation', 'hter']]
test = test[['index', 'original', 'translation']]

index = test['index'].to_list()
train = train.rename(columns={'original': 'text_a', 'translation': 'text_b', 'hter': 'labels'}).dropna()
dev = dev.rename(columns={'original': 'text_a', 'translation': 'text_b', 'hter': 'labels'}).dropna()
dev_test = dev_test.rename(columns={'original': 'text_a', 'translation': 'text_b', 'hter': 'labels'}).dropna()
test = test.rename(columns={'original': 'text_a', 'translation': 'text_b'}).dropna()

train = fit(train, 'labels')
dev = fit(dev, 'labels')
dev_test = fit(dev_test, 'labels')
test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))
assert (len(index) == 2412)
if monotransquest_config["evaluate_during_training"]:
    if monotransquest_config["n_fold"] > 1:
        dev_preds = np.zeros((len(dev), monotransquest_config["n_fold"]))
        dev_test_preds = np.zeros((len(dev_test), monotransquest_config["n_fold"]))
        test_preds = np.zeros((len(test), monotransquest_config["n_fold"]))
        for i in range(monotransquest_config["n_fold"]):

            if os.path.exists(monotransquest_config['output_dir']) and os.path.isdir(
                    monotransquest_config['output_dir']):
                shutil.rmtree(monotransquest_config['output_dir'])

            model = MonoTransQuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
                                        args=monotransquest_config)
            train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * i)
            model.train_model(train_df, eval_df=eval_df, pearson_corr=pearson_corr, spearman_corr=spearman_corr,
                              mae=mean_absolute_error)
            model = MonoTransQuestModel(MODEL_TYPE, monotransquest_config["best_model_dir"], num_labels=1,
                                        use_cuda=torch.cuda.is_available(), args=monotransquest_config)
            result, model_outputs, wrong_predictions = model.eval_model(dev, pearson_corr=pearson_corr,
                                                                        spearman_corr=spearman_corr,
                                                                        mae=mean_absolute_error)
            result_test, model_outputs_test, wrong_predictions_test = model.eval_model(dev_test, pearson_corr=pearson_corr,
                                                                        spearman_corr=spearman_corr, mae=mean_absolute_error)
            predictions, raw_outputs = model.predict(test_sentence_pairs)
            dev_preds[:, i] = model_outputs
            test_preds[:, i] = predictions
            dev_test_preds[:, i] = model_outputs_test

        dev['predictions'] = dev_preds.mean(axis=1)
        test['predictions'] = test_preds.mean(axis=1)
        dev_test['predictions'] = dev_test_preds.mean(axis=1)

    else:
        model = MonoTransQuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
                                    args=monotransquest_config)
        train, eval_df = train_test_split(train, test_size=0.1, random_state=SEED)
        model.train_model(train, eval_df=eval_df, pearson_corr=pearson_corr, spearman_corr=spearman_corr,
                          mae=mean_absolute_error)
        model = MonoTransQuestModel(MODEL_TYPE, monotransquest_config["best_model_dir"], num_labels=1,
                                    use_cuda=torch.cuda.is_available(), args=monotransquest_config)
        result, model_outputs, wrong_predictions = model.eval_model(dev, pearson_corr=pearson_corr,
                                                                    spearman_corr=spearman_corr,
                                                                    mae=mean_absolute_error)
        result_test, model_outputs_test, wrong_predictions_test = model.eval_model(dev_test, pearson_corr=pearson_corr,
                                                                    spearman_corr=spearman_corr,
                                                                    mae=mean_absolute_error)
        predictions, raw_outputs = model.predict(test_sentence_pairs)
        dev['predictions'] = model_outputs
        dev_test['predictions'] = model_outputs_test
        test['predictions'] = predictions

else:
    model = MonoTransQuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
                                args=monotransquest_config)
    model.train_model(train, pearson_corr=pearson_corr, spearman_corr=spearman_corr, mae=mean_absolute_error)
    result, model_outputs, wrong_predictions = model.eval_model(dev, pearson_corr=pearson_corr,
                                                                spearman_corr=spearman_corr, mae=mean_absolute_error)
    result_test, model_outputs_test, wrong_predictions_test = model.eval_model(dev_test, pearson_corr=pearson_corr,
                                                                spearman_corr=spearman_corr, mae=mean_absolute_error)
    predictions, raw_outputs = model.predict(test_sentence_pairs)
    dev['predictions'] = model_outputs
    dev_test['predictions'] = model_outputs_test
    test['predictions'] = predictions

dev = un_fit(dev, 'labels')
dev = un_fit(dev, 'predictions')
test = un_fit(test, 'predictions')
dev_test = un_fit(dev_test, 'predictions')
dev.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')
dev_test.to_csv(os.path.join(TEMP_DIRECTORY, 'devtest.tsv'), header=True, sep='\t', index=False, encoding='utf-8')
test.to_csv(os.path.join(TEMP_DIRECTORY, 'test.tsv'), header=True, sep='\t', index=False, encoding='utf-8')
# test.to_csv(os.path.join(TEMP_DIRECTORY, 'test_eval.tsv'), header=True, sep='\t', index=False, encoding='utf-8')
draw_scatterplot(dev, 'labels', 'predictions', os.path.join(TEMP_DIRECTORY, RESULT_IMAGE), "Chinese-English")
draw_scatterplot(dev_test, 'labels', 'predictions', os.path.join(TEMP_DIRECTORY, "dev_test.jpg"), "Chinese-English")
print_stat(dev, 'labels', 'predictions')
print_stat(dev_test, 'labels', 'predictions')
format_submission(df=test, index=index, language_pair="zh-en", method="TransQuest",
                  path=os.path.join(TEMP_DIRECTORY, SUBMISSION_FILE))