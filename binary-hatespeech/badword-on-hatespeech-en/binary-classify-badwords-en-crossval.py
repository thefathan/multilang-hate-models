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
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler

# hs_class is 'negative' WITHOUT SEXUAL WORD 
neg_df1 = pd.read_csv('/nas.dbms/fathan/test/multilang-hate-models/binary-hatespeech/badword-on-hatespeech-en/negative_hatespeech_without_sexual_words.csv')

# hs_class is 'negative' WITH SEXUAL WORD 
neg_df = pd.read_csv('/nas.dbms/fathan/test/multilang-hate-models/binary-hatespeech/badword-on-hatespeech-en/negative_hatespeech_with_sexual_words.csv')

# hs_class is 'positive' WITH SEXUAL WORD 
pos_df1 = pd.read_csv('/nas.dbms/fathan/test/multilang-hate-models/binary-hatespeech/badword-on-hatespeech-en/positive_hatespeech_with_sexual_words.csv')

# hs_class is 'positive' WITHOUT SEXUAL WORD 
pos_df = pd.read_csv('/nas.dbms/fathan/test/multilang-hate-models/binary-hatespeech/badword-on-hatespeech-en/positive_hatespeech_without_sexual_words.csv')

# # hs_class is 'positive' WITH SEXUAL WORD + WITHOUT SEXUAL WORD 25%
# frac_sample = 0.25
# pos_df1 = (pd.read_csv('/nas.dbms/fathan/test/multilang-hate-models/binary-hatespeech/badword-on-hatespeech-en/positive_hatespeech_with_sexual_words.csv')).sample(frac=frac_sample, random_state=42)
# pos_df1 = pd.read_csv('/nas.dbms/fathan/test/multilang-hate-models/binary-hatespeech/badword-on-hatespeech-en/positive_hatespeech_with_sexual_words.csv')
# pos_df2 = (pd.read_csv('/nas.dbms/fathan/test/multilang-hate-models/binary-hatespeech/badword-on-hatespeech-en/positive_hatespeech_without_sexual_words.csv')).sample(frac=frac_sample, random_state=42)
# pos_df2 = pd.read_csv('/nas.dbms/fathan/test/multilang-hate-models/binary-hatespeech/badword-on-hatespeech-en/positive_hatespeech_without_sexual_words.csv')

# pos_df = pd.concat([pos_df1, pos_df2], ignore_index=True)

# Ensure both DataFrames have the correct column names
neg_df.columns = ['text', 'hs_class']
pos_df.columns = ['text', 'hs_class']
neg_df1.columns = ['text', 'hs_class']
pos_df1.columns = ['text', 'hs_class']

df = pd.concat([neg_df1, pos_df], ignore_index=True)
# Load the dataset
val_df = pd.read_csv('/nas.dbms/fathan/test/multilang-hate-models/binary-hatespeech/badword-on-hatespeech-en/data_preprocessed_en_translated.csv')

# Keep only the relevant columns (English text and hs_class)
val_df = val_df[['text_translated', 'hs_class']]

# Rename columns to match the desired format
val_df = val_df.rename(columns={'text_translated': 'text'})

val_df = val_df.sample(frac=0.2, random_state=42)


# val_df = pd.concat([neg_df1, pos_df1], ignore_index=True)
# val_df = val_df.sample(frac=0.2, random_state=42)


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

# Map the 'hs_class' values back to integers for the model
df['hs_class'] = df['hs_class'].map({'positive': 1, 'negative': 0})
val_df['hs_class'] = val_df['hs_class'].map({'positive': 1, 'negative': 0})

# Undersample using imbalanced-learn
# Create the undersampling object
undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)

# Separate the features (X) and the target (y)
X = df['text'].values.reshape(-1, 1)  # Reshape required because RandomUnderSampler expects a 2D array
y = df['hs_class']
X1 = val_df['text'].values.reshape(-1, 1)  # Reshape required because RandomUnderSampler expects a 2D array
y1 = val_df['hs_class']


# Perform the undersampling
X_resampled, y_resampled = undersampler.fit_resample(X, y)
X1_resampled, y1_resampled = undersampler.fit_resample(X1, y1)

# Create a new dataframe with the resampled data
df = pd.DataFrame({'text': X_resampled.flatten(), 'hs_class': y_resampled}) # df here is already resampled
val_df = pd.DataFrame({'text': X1_resampled.flatten(), 'hs_class': y1_resampled}) # val_df here is already resampled

# Display the modified DataFrame
print(df.info())
print(df.sample(5))
print(df['hs_class'].value_counts())

# Split the dataset
train_df = df
test_df = val_df

# Convert pandas DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Rename 'hs_class' to 'labels' for the model input
train_dataset = train_dataset.rename_column('hs_class', 'labels')
test_dataset = test_dataset.rename_column('hs_class', 'labels')

# Remove columns that are not input for the model (keep 'labels')
# Remove only 'text' since '__index_level_0__' is not present in the dataset anymore
train_dataset = train_dataset.remove_columns(['text'])
test_dataset = test_dataset.remove_columns(['text'])

# Set format for PyTorch
train_dataset.set_format('torch')
test_dataset.set_format('torch')

# Create a data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=1000,
    evaluation_strategy="epoch", # Evaluate at the end of each epoch
    save_steps=10000,  # Save a checkpoint every 10000 steps (adjust as needed)
    save_total_limit=2,  # Keep only the 2 most recent checkpoints
    learning_rate=1e-6  # Custom learning rate
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
