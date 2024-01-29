# Import necessary libraries
import pandas as pd
import numpy as np
import torch
import logging
import argparse

from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from peft import PeftModel, PeftConfig
import sacrebleu

import data_processing

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def perform_inference(test_df, peft_model, tokenizer, batch_size=8):
    preds = []
    for i in range(0, len(test_df), batch_size):
        batch_texts = test_df.iloc[i:i+batch_size]['input_text'].tolist()
        tokenized_inputs = tokenizer(batch_texts, return_tensors="pt", max_length=100, truncation=True, padding="max_length")

        input_ids = tokenized_inputs['input_ids'].to(device)
        attention_mask = tokenized_inputs['attention_mask'].to(device)

        # Generate predictions in batches
        outputs = peft_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=100, num_beams=5, repetition_penalty=1.5, early_stopping=True)
        
        # Decode each output in the batch and add to the predictions list
        preds.extend([tokenizer.decode(output, skip_special_tokens=True) for output in outputs])
    
    test_df["preds"] = preds
    return test_df

def main(model_name, json_file_path):
    # Initialize the tokenizer and model dynamically based on the model_name argument
    #tokenizer = MT5Tokenizer.from_pretrained(model_name)
    #model = MT5ForConditionalGeneration.from_pretrained(model_name).to(device)

    tokenizer = MT5Tokenizer.from_pretrained('google/mt5-large')
    model = MT5ForConditionalGeneration.from_pretrained('google/mt5-large')

    peft_model = PeftModel.from_pretrained(model,
                                       '/content/drive/MyDrive/swiss-german-normalization/mt5-large_peft/best_model',
                                       is_trainable=False)

    merged_model = peft_model.merge_and_unload()

    merged_model.to(device)
    # Use the refactored data_processing module to get the test data
    _, _, test_df = data_processing.get_data_splits(json_file_path)

    # Perform inference
    result_df = perform_inference(test_df, merged_model, tokenizer)
    return result_df

if __name__ == "__main__":
    # Parse command-line arguments for model size
    parser = argparse.ArgumentParser(description='mT5 model size selector')
    parser.add_argument('model_name', type=str, help='Model size (e.g., google/mt5-small, google/mt5-base, google/mt5-large, etc.)')
    args = parser.parse_args()

    # Path to your JSON file containing the test data
    json_file_path = '/content/swiss-german-normalization/sentences_ch_de_numerics.json'

    # Call the main function to perform inference with the specified model size
    result_df = main(args.model_name, json_file_path)

    # Calculate ChrF++ scores
    chrf_scores = []
    for index, row in result_df.iterrows():
        prediction = row['preds']
        reference = [row['target_text']]
        score = sacrebleu.sentence_chrf(prediction, reference, beta=2, word_order=1).score
        chrf_scores.append(score)

    scores = sum(chrf_scores) / len(chrf_scores)
    
    print('')
    print('ChrF++ score:') 
    print(scores) 
    print('')

    # Optional: Save the result_df to a CSV file
    output_folder = '/content/drive/MyDrive/swiss-german-normalization/'
    result_df.to_csv(output_folder + 'test_predictions_' + args.model_name.replace("/", "_") + '.csv', index=False)
    logger.info("Inference completed and saved to CSV.")
