from multiprocessing import cpu_count

SEED = 777
TEMP_DIRECTORY = "/yh/TransQuest-master/examples/sentence_level/wmt_2019/temp/data"
RESULT_FILE = "result.tsv"
SUBMISSION_FILE = "predictions.txt"
RESULT_IMAGE = "result.jpg"
GOOGLE_DRIVE = False
DRIVE_FILE_ID = None
MODEL_TYPE = "xlmroberta"
MODEL_NAME = "xlm-roberta-large"

monotransquest_config = {
    'output_dir': '/home/yljiang/code/qe/res/zhen/temp/outputs/',
    "best_model_dir": "/home/yljiang/code/qe/res/zhentemp/outputs/best_model",
    'cache_dir': '/home/yljiang/code/qe/res/zhentemp/cache_dir/',

    'fp16': True,
    'fp16_opt_level': 'O1',
    'max_seq_length': 250,
    'train_batch_size': 8,
    'gradient_accumulation_steps': 1,
    'eval_batch_size': 8,
    'num_train_epochs': 5,
    'weight_decay': 0.0001,
    'learning_rate': 1e-5,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0.1,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,
    'do_lower_case': False,

    'logging_steps': 500,
    'save_steps': 1000,
    "no_cache": False,
    'save_model_every_epoch': True,
    'n_fold': 3,
    'evaluate_during_training': True,
    'evaluate_during_training_steps': 1000,
    "evaluate_during_training_verbose": True,
    'use_cached_eval_features': False,
    'save_eval_checkpoints': True,
    'tensorboard_dir': None,

    'regression': True,

    'overwrite_output_dir': True,
    'reprocess_input_data': True,

    'process_count': 1 if cpu_count() > 2 else 1,
    'n_gpu': 1,
    'use_multiprocessing': True,
    'silent': False,

    'wandb_project': None,
    'wandb_kwargs': {},

    "use_early_stopping": True,
    "early_stopping_patience": 10,
    "early_stopping_delta": 0,
    "early_stopping_metric": "eval_loss",
    "early_stopping_metric_minimize": True,

    "manual_seed": 777,

    "encoding": None,
}
