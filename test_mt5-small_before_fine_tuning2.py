# Import necessary libraries
import pandas as pd
import numpy as np
import torch
import logging
from transformers import MT5ForConditionalGeneration, T5Tokenizer
import sacrebleu

# Assuming your data_processing module works as expected with transformers
import data_processing

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def perform_inference(test_df, model, tokenizer):
    # Prepare the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    preds = []
    for input_text in test_df.input_text.values.tolist():
        # Encode the input text
        encoded = tokenizer.encode_plus(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        # Generate prediction
        output = model.generate(input_ids, attention_mask=attention_mask, max_length=100, num_beams=5, length_penalty=2.5, repetition_penalty=1.5, early_stopping=True)
        
        # Decode the generated ids
        pred_text = tokenizer.decode(output[0], skip_special_tokens=True)
        preds.append(pred_text)
    
    test_df["preds"] = preds
    return test_df

def main(model, tokenizer, json_file_path):
    # Use the refactored data_processing module to get the test data
    _, _, test_df = data_processing.get_data_splits(json_file_path)

    # Perform inference
    result_df = perform_inference(test_df, model, tokenizer)
    return result_df

if __name__ == "__main__":
    # Initialize the tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")

    # Path to your JSON file containing the test data
    json_file_path = '/content/swiss-german-normalization/sentences_ch_de_numerics.json'

    # Call the main function to perform inference
    result_df = main(model, tokenizer, json_file_path)

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
    result_df.to_csv(output_folder + 'test_predictions_mt5-small_before_finetuning.csv', index=False)
    logger.info("Inference completed and saved to CSV.")
