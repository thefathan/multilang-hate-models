# Install required packages
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import datasets
except ImportError:
    install('datasets')
    import datasets

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, BertTokenizer, BertForSequenceClassification, RobertaForSequenceClassification, XLMRobertaForSequenceClassification
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('/nas.dbms/fathan/test/multilang-hate-models/data_preprocessed.csv', header=0)
df['label'] = df['hs_class'].map({'positive': 1, 'negative': 0})

# Ensure 'text' column contains strings
df['text'] = df['text'].astype(str)

# Split the dataset
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# # Load the tokenizer and model (IndoBERT)
# model_name = 'indolem/indobert-base-uncased'
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# # Load the tokenizer and model (BERT)
# model_name = 'bert-base-uncased'
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# # Load the tokenizer and model (RoBERTa)
# model_name = 'roberta-base'
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Load the tokenizer and model (XLM-RoBERTa)
model_name = 'FacebookAI/xlm-roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Remove columns that are not input for the model
train_dataset = train_dataset.remove_columns(['text', 'hs_class', '__index_level_0__'])
test_dataset = test_dataset.remove_columns(['text', 'hs_class', '__index_level_0__'])

# Set format for PyTorch
train_dataset.set_format('torch')
test_dataset.set_format('torch')

# Create a data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=20,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=1000,
    evaluation_strategy="epoch",
    learning_rate=1e-5  # Custom learning rate
)

# Define the compute_metrics function
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator  # Use the data collator for padding
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()

# Print evaluation results
print(f"Evaluation results:\n"
      f"  Loss: {eval_results['eval_loss']:.4f}\n"
      f"  Accuracy: {eval_results['eval_accuracy']:.4f}\n"
      f"  Precision: {eval_results['eval_precision']:.4f}\n"
      f"  Recall: {eval_results['eval_recall']:.4f}\n"
      f"  F1 Score: {eval_results['eval_f1']:.4f}")

# Collect evaluation metrics over epochs
history = trainer.state.log_history
f1_scores = [x['eval_f1'] for x in history if 'eval_f1' in x]

# Plot F1 Score over epochs
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(f1_scores) + 1), f1_scores, marker='o', linestyle='-', color='b')
plt.title('F1 Score over Epochs')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.grid(True)
plt.savefig('f1_score_over_epochs.png')
plt.show()

# Generate predictions for confusion matrix
predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)
labels = predictions.label_ids

# Confusion Matrix
cm = confusion_matrix(labels, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Hate Speech', 'Hate Speech'], yticklabels=['Not Hate Speech', 'Hate Speech'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()
