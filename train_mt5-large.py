import logging
from simpletransformers.t5 import T5Model, T5Args
import data_processing

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

def get_model_args():
    model_args = T5Args()
    model_args.max_seq_length = 100
    model_args.train_batch_size = 5
    model_args.eval_batch_size = 5
    model_args.num_train_epochs = 10
    model_args.scheduler = "cosine_schedule_with_warmup"
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_steps = 10000
    model_args.learning_rate = 0.0003
    model_args.optimizer = 'Adafactor'
    model_args.use_multiprocessing = False
    model_args.fp16 = False
    model_args.save_steps = -1
    model_args.save_eval_checkpoints = True
    model_args.no_cache = True
    model_args.reprocess_input_data = True
    model_args.overwrite_output_dir = True
    model_args.save_model_every_epoch = False
    model_args.preprocess_inputs = False
    model_args.use_early_stopping = True
    model_args.num_return_sequences = 1
    model_args.do_lower_case = True
    model_args.output_dir = "/content/drive/MyDrive/swiss-german-normalization/mT5-large/"
    model_args.best_model_dir = "/content/drive/MyDrive/swiss-german-normalization/mT5-large/best_model"
    model_args.wandb_project = "CH-DE mT5-large"

    return model_args

def fine_tune_t5(train_df, val_df, model_args):
    model = T5Model("mt5", "google/mt5-large", args=model_args)
    model.train_model(train_df, eval_data=val_df)

def main():
    setup_logging()

    model_args = get_model_args()

    train_df, val_df, _ = data_processing.get_data_splits('/content/swiss-german-normalization/sentences_ch_de_numerics.json')

    fine_tune_t5(train_df, val_df, model_args)

if __name__ == "__main__":
    main()
