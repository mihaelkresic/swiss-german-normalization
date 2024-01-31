import argparse
import logging
import data_processing
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from torch.utils.data import Dataset
import torch

#new
#from torch_optimizer import Adafactor
from transformers.optimization import Adafactor, AdafactorSchedule

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Custom dataset class
class MT5Dataset(Dataset):
    def __init__(self, tokenizer, input_texts, target_texts, max_length=100):
        self.tokenizer = tokenizer
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.max_length = max_length

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        source = self.input_texts[idx]
        target = self.target_texts[idx]
        source_encodings = self.tokenizer(source, max_length=self.max_length, padding='max_length', truncation=True)
        target_encodings = self.tokenizer(target, max_length=self.max_length, padding='max_length', truncation=True)
        return {
          "input_ids": source_encodings["input_ids"], 
          "attention_mask": source_encodings["attention_mask"], 
          "labels": target_encodings["input_ids"]}

def main(model_size):
    run_name = f"{model_size}"
    model_name = f"google/{model_size}"
    output_dir = f"/content/drive/MyDrive/swiss-german-normalization/{model_size}/"
  
    # Load the tokenizer and model
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name).to(device)


    # Path to your JSON file containing the test data
    json_file_path = '/content/swiss-german-normalization/sentences_ch_de_numerics.json'
  
    # Load and process data
    train_df, val_df, _ = data_processing.get_data_splits(json_file_path)
  
    train_dataset = MT5Dataset(tokenizer, train_df['input_text'].tolist(), train_df['target_text'].tolist())
    val_dataset = MT5Dataset(tokenizer, val_df['input_text'].tolist(), val_df['target_text'].tolist())

    #new 1: Adafactor + 5e-5
    optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=1e-3)
    lr_scheduler = None

    
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,                # Output directory for model checkpoints
        overwrite_output_dir=True,            # Overwrite the content of the output dir
        logging_dir=None,
        per_device_train_batch_size=8,        # Batch size for training
        per_device_eval_batch_size=8,         # Batch size for evaluation
        #learning_rate=5e-5,
        learning_rate=1e-3,
        #new
        lr_scheduler_type="constant_with_warmup",
        num_train_epochs=10,                  # Number of training epochs
        warmup_steps=500,                     # Number of warmup steps for learning rate scheduler
        evaluation_strategy="steps",          # Evaluation strategy
        save_strategy="steps",                # Save strategy
        do_eval=True,
        save_steps=2000,                       # Save checkpoint every X steps
        eval_steps=2000,                       # Evaluate model every X steps
        save_total_limit=2,
        predict_with_generate=True,            # Use generate for prediction
        load_best_model_at_end=True,           # Load the best model at the end of training
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        run_name=run_name
    )

    # Initialize the trainerb
    trainer = Seq2SeqTrainer(
        model=model,
        optimizers=(optimizer, lr_scheduler),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer)
    )

    # Train the model
    trainer.train()

    best_model_dir = f"/content/drive/MyDrive/swiss-german-normalization/{model_size}/best_model_/"
    model.save_pretrained(best_model_dir)

    best_model_dir2 = f"/content/drive/MyDrive/swiss-german-normalization/{model_size}/best_model/"
    trainer.save_model(best_model_dir2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train mT5 model')
    parser.add_argument('model_size', type=str, help='Model size (e.g., mt5-small, mt5-base, etc.)')
    args = parser.parse_args()
    main(args.model_size)
