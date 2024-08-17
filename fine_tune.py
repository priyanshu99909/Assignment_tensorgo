from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# 1. Load Dataset
dataset = load_dataset('imdb')

# 2. Load Pre-trained Model and Tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 3. Data Preprocessing
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True)

encoded_dataset = dataset.map(preprocess_function, batched=True)
encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# 4. Training Arguments
training_args = TrainingArguments(
    output_dir='./results',               # Required argument
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy='epoch'
)

# 5. Evaluation Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=-1)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {'accuracy': accuracy, 'f1': f1}

# 6. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['test'],
    compute_metrics=compute_metrics
)

# 7. Fine-Tune the Model (Mocked for Explanation)
# trainer.train()

# 8. Evaluate the Model
# results = trainer.evaluate()
# print("Evaluation Results:", results)

# 9. Save the Fine-Tuned Model (Mocked for Explanation)
# model.save_pretrained('./fine-tuned-model')
# tokenizer.save_pretrained('./fine-tuned-model')

# 10. Testing the Model with a Sample Input (Optional)
def predict(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = np.argmax(logits.detach().numpy())
    return predicted_class

sample_text = "I love this movie.It is so amazing."
predicted_class = predict(sample_text)
print("Sample text prediction:", "Positive" if predicted_class == 1  else "Negative")

