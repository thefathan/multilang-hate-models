# Install required packages
import subprocess
import sys
import os

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
from imblearn.under_sampling import RandomUnderSampler

# Load the dataset
df = pd.read_csv('/nas.dbms/fathan/test/multilang-hate-models/binary-hatespeech/badword-on-hatespeech-en/multilang-cross-val/id_dataset_huggingface1/id_huggingface1_formatted.csv')

# Ensure 'text' column contains strings
df['text'] = df['text'].astype(str)

# Map the 'hs_class' values back to integers for the model
df['hs_class'] = df['hs_class'].map({'positive': 1, 'negative': 0})

# Undersample using imbalanced-learn
# Create the undersampling object
undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)

# Separate the features (X) and the target (y)
X = df['text'].values.reshape(-1, 1)  # Reshape required because RandomUnderSampler expects a 2D array
y = df['hs_class']

# Perform the undersampling
X_resampled, y_resampled = undersampler.fit_resample(X, y)

# Create a new dataframe with the resampled data
df = pd.DataFrame({'text': X_resampled.flatten(), 'hs_class': y_resampled}) # df here is already resampled
print(df['hs_class'].value_counts())


# Load the tokenizer and model (IndoBERT)
model_name = 'indolem/indobert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# # Load the tokenizer and model (BERT)
# model_name = 'bert-base-uncased'
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# # Load the tokenizer and model (RoBERTa)
# model_name = 'roberta-base'
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

# # Load the tokenizer and model (XLM-RoBERTa)
# model_name = 'FacebookAI/xlm-roberta-base'
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize the dataset with padding and truncation
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',   # Ensure sequences are padded to the same length
        truncation=True,
        max_length=512
    )

# Modify the train-test split to train-validation-test (0.7, 0.15, 0.15)
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)  # Split remaining 30% equally for validation and test

# Map 'hs_class' to 'labels' as integers for each split (train, val, test)
train_df['labels'] = train_df['hs_class'].astype(int)
val_df['labels'] = val_df['hs_class'].astype(int)
test_df['labels'] = test_df['hs_class'].astype(int)

# Convert pandas DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# Apply the tokenization to the datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Remove unnecessary columns, keeping 'labels' and input data
train_dataset = train_dataset.remove_columns(['text', 'hs_class', '__index_level_0__'])
val_dataset = val_dataset.remove_columns(['text', 'hs_class', '__index_level_0__'])
test_dataset = test_dataset.remove_columns(['text', 'hs_class', '__index_level_0__'])

# Set the format for PyTorch tensors
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Create a data collator that dynamically pads the input sequences during training
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Reduce batch size to fit within memory constraints
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=20,
    per_device_train_batch_size=8,   # Reduced batch size to lower memory usage
    per_device_eval_batch_size=8,    # Reduced batch size for evaluation
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=1000,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=1,
    learning_rate=1e-5,
    fp16=True,  # Enable mixed precision for memory optimization
)

# Enable gradient checkpointing to save memory during training
model.gradient_checkpointing_enable()

# Set environment variable to control memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

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

# Initialize the Trainer with memory-optimized settings
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

# Train the model
trainer.train()

# Evaluate the model on the test dataset
eval_results = trainer.evaluate(test_dataset)


# Print evaluation results on the test set
print(f"Evaluation results on test set:\n"
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
