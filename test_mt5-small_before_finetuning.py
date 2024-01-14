# Import necessary modules
import pandas as pd
import numpy as np

import torch

import logging
from simpletransformers.t5 import T5Model, T5Args

#from rouge import Rouge
import sacrebleu

# If data_processing is in a different directory, you might need to append the directory to sys.path
# import sys
# sys.path.append('/path/to/directory/where/data_processing.py/is')

# Now, import your refactored data_processing module
import data_processing

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def perform_inference(test_df, model):
    # Perform the inference and add a new column with predictions to the DataFrame
    test_df["preds"] = model.predict(test_df.input_text.values.tolist())
    return test_df

def main(model, json_file_path):
    # Use the refactored data_processing module to get the test data
    _, _, test_df = data_processing.get_data_splits(json_file_path)

    # Perform inference
    result_df = perform_inference(test_df, model)
    return result_df

if __name__ == "__main__":
    # Define model arguments
    model_args = T5Args()
    model_args.max_length = 100
    model_args.length_penalty = 2.5
    model_args.repetition_penalty = 1.5
    model_args.num_beams = 5

    # Initialize the model
    model = T5Model("mt5", "google/mt5-small", args=model_args)

    # Path to your JSON file containing the test data
    json_file_path = '/content/swiss-german-normalization/sentences_ch_de_numerics.json'

    # Call the main function to perform inference
    result_df = main(model, json_file_path)

    
    # Initialize the ROUGE metric
    #rouge = Rouge()

    # Prepare the data for ROUGE calculation
    # Convert the 'preds' and 'target_text' columns to lists of strings
    #predictions = result_df["preds"].values.tolist()
    #references = result_df["target_text"].values.tolist()

    # Calculate ROUGE scores
    #scores = rouge.get_scores(predictions, references, avg=True)

    predictions = result_df["preds"].values.tolist()
    references = result_df["target_text"].apply(lambda x: [x]).values.tolist()

    # Calculate ChrF++ scores
    scores = sacrebleu.corpus_chrf(predictions, references, beta=2, word_order=1).score
    
    print('')
    print('ChrF++ score: ') 
    print(scores) 
    print('')

    # Optional: Save the result_df to a CSV file
    output_folder = '/content/drive/MyDrive/swiss-german-normalization/'
    result_df.to_csv(output_folder + 'test_predictions_mt5-small_before_finetuning.csv', index=False)
    logger.info("Inference completed and saved to CSV.")
