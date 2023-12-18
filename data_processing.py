import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Clean up the text by lowercasing the text, stripping trailing spaces and removing quotation marks
def clean_text(text):
    # Replace different types of quotation marks with nothing
    replacements = ['\"', '«', '»', '« ', ' »', '“', '”', '"']
    for r in replacements:
        text = text.replace(r, '')

    # Strip spaces and lowercase the text
    text = text.strip().lower()
    return text

# Step 1: Parse the JSON File
json_file_path = '/content/swiss-german-normalization/sentences_ch_de_numerics.json'
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Extract and format sentence pairs
sentence_pairs = []
for item in data:
    high_german = clean_text(item['de'])
    for key in item:
        if key.startswith('ch_'):
            swiss_german = clean_text(item[key])
            sentence_pairs.append({'input_text': swiss_german, 'target_text': high_german, 'prefix': ''})

# Convert to DataFrame
df = pd.DataFrame(sentence_pairs)

# Step 2: Data Splitting
# Split the data into 80% training, 10% validation, and 10% testing
train_df, test_df = train_test_split(df, test_size=0.2, random_state=93)
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=93)

# Step 3: Saving the Data
output_folder = '/content/drive/MyDrive/swiss-german-normalization/'
train_df.to_csv(output_folder + 'train.csv', index=False)
val_df.to_csv(output_folder + 'val.csv', index=False)
test_df.to_csv(output_folder + 'test.csv', index=False)
