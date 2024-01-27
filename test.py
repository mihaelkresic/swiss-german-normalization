# Import necessary libraries
import pandas as pd
import numpy as np
import torch
import logging
import argparse

from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import sacrebleu

import data_processing

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def perform_inference(test_df, model, tokenizer):

    preds = []
    for input_text in test_df.input_text.values.tolist():
        # Encode the input text

        #input_text = "translate German to English: " + original_text
        
        tokenized_input = tokenizer(input_text, return_tensors="pt", max_length=90, truncation=True, padding="max_length")

        input_ids = tokenized_input['input_ids'].to(device)
        attention_mask = tokenized_input['attention_mask'].to(device)

        # Generate prediction
        #output = model.generate(input_ids=input["input_ids"], attention_mask=input["attention_mask"])
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=5, length_penalty=2.5, repetition_penalty=1.5, early_stopping=True)
        
        
        # Decode the generated ids
        pred_text = tokenizer.decode(output[0], skip_special_tokens=True)
        preds.append(pred_text)
    
    test_df["preds"] = preds
    return test_df

def main(model_name, json_file_path):
    # Initialize the tokenizer and model dynamically based on the model_name argument
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name).to(device)

    # Use the refactored data_processing module to get the test data
    _, _, test_df = data_processing.get_data_splits(json_file_path)

    # Perform inference
    result_df = perform_inference(test_df, model, tokenizer)
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
