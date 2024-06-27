import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

model_path = 'trained_t5_model.pkl'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the checkpoint
checkpoint = torch.load(model_path, map_location=device)

# Inspect the keys in the checkpoint
# print("Checkpoint keys:", checkpoint.keys())

# Assuming the model's state dictionary key is 'state_dict' or any other key from the printed keys
model_state_dict_key = 'model_state_dict'  # Change this based on the actual key in the checkpoint
if model_state_dict_key not in checkpoint:
    raise KeyError(f"Key '{model_state_dict_key}' not found in the checkpoint. Available keys: {list(checkpoint.keys())}")

model_state_dict = checkpoint[model_state_dict_key]

# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')
model.load_state_dict(model_state_dict)

# Set the model to evaluation mode
model.eval()

print(type(model))