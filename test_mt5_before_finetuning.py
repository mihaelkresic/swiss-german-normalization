from simpletransformers.t5 import T5Model, T5Args

model_args = T5Args()
model_args.max_length = 100
model_args.length_penalty = 2.5
model_args.repetition_penalty = 1.5
model_args.num_beams = 5

model = T5Model("mt5", "google/mt5-small", args=model_args)

#Perform the inference
test_df["preds"] = model.predict(test_df.input_text.values.tolist())
