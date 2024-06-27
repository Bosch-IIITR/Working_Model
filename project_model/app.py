from flask import Flask, request, jsonify, render_template
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = Flask(__name__)


model_path = 'trained_t5_model.pkl'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = T5ForConditionalGeneration.from_pretrained('t5-small')
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

tokenizer = T5Tokenizer.from_pretrained('t5-small')

def generate_answer(question):
    input_text = "question: %s </s>" % question
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512)
    outputs = model.generate(input_ids.to(device), max_length=100, num_beams=4, length_penalty=2.0, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    answer = generate_answer(question)
    return jsonify({'answer': answer})

if __name__ == "__main__":
    app.run(debug=True)
