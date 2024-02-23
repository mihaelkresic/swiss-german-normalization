import pandas as pd
import numpy as np
import torch
import logging
import argparse

from transformers import MT5ForConditionalGeneration, MT5Tokenizer

import sacrebleu
from comet import download_model, load_from_checkpoint

import data_processing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def perform_inference(test_df, model, tokenizer, batch_size=8):
    preds = []
    for i in range(0, len(test_df), batch_size):
        batch_texts = test_df.iloc[i:i+batch_size]['input_text'].tolist()
        tokenized_inputs = tokenizer(batch_texts, return_tensors="pt", max_length=100, truncation=True, padding="max_length")

        input_ids = tokenized_inputs['input_ids'].to(device)
        attention_mask = tokenized_inputs['attention_mask'].to(device)

        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=100, num_beams=5, repetition_penalty=1.5, early_stopping=True)
        
        preds.extend([tokenizer.decode(output, skip_special_tokens=True) for output in outputs])
    
    test_df["preds"] = preds
    return test_df

def main(model_name, json_file_path):
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name).to(device)

    _, _, test_df = data_processing.get_data_splits(json_file_path)

    # Perform inference
    result_df = perform_inference(test_df, model, tokenizer)
    return result_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='mT5 model size selector')
    parser.add_argument('model_name', type=str, help='Model size (e.g., google/mt5-small, google/mt5-base, google/mt5-large, etc.)')
    args = parser.parse_args()

    json_file_path = '/content/swiss-german-normalization/sentences_ch_de_numerics.json'

    result_df = main(args.model_name, json_file_path)

    chrf_scores = []
    for index, row in result_df.iterrows():
        prediction = row['preds']
        reference = [row['target_text']]
        score = sacrebleu.sentence_chrf(prediction, reference).score
        chrf_scores.append(score)

    scores = sum(chrf_scores) / len(chrf_scores)
    
    print('')
    print('ChrF score:') 
    print(scores) 
    print('')

    comed_model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comed_model_path)

    data = [{"src": row["input_text"], "mt": row["preds"], "ref": row["target_text"]} for index, row in result_df.iterrows()]

    comet_scores = comet_model.predict(data, batch_size=8, gpus=1)

    print('')
    print('COMET score:') 
    print(comet_scores.system_score) 
    print('')

    output_folder = '/content/drive/MyDrive/swiss-german-normalization/'
    result_df.to_csv(output_folder + 'test_predictions_' + args.model_name.replace("/", "_") + '.csv', sep=";", index=False, encoding='utf-8-sig')
    logger.info("Inference completed and saved to CSV.")
