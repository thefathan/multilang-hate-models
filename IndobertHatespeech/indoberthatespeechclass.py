import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import IntervalStrategy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Load dataset
df = pd.read_csv('/nas.dbms/fathan/test/multilang-hate-models/data_preprocessed.csv', header=0)
df.info()
print(df.sample(5))

# Data Preprocessing
df.dropna(subset=['text'], inplace=True)
label_mapping = {"positive": 1, "negative": 0}
df['label'] = df['hs_class'].map(label_mapping)
df.reset_index(drop=True, inplace=True)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")

# Define k-fold cross-validation
k_folds = 3
skf = StratifiedKFold(n_splits=k_folds)

def preprocess_data(texts, labels):
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt', max_length=512)
    dataset = torch.utils.data.TensorDataset(encodings['input_ids'], encodings['attention_mask'], torch.tensor(labels))
    return dataset

metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
training_losses = []
validation_losses = []
validation_losses_per_epoch = []

class CustomCallback(TrainerCallback):
    def __init__(self, patience=2, factor=0.5, min_lr=1e-6):
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.best_loss = float('inf')
        self.wait = 0

    def on_evaluate(self, args, state, control, **kwargs):
        logs = kwargs.get('logs', {})
        current_loss = logs.get('eval_loss')
        if current_loss is not None:
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = self.trainer.args.learning_rate
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    if old_lr != new_lr:
                        print(f"\nReducing learning rate from {old_lr} to {new_lr}")
                        self.trainer.args.learning_rate = new_lr
                    self.wait = 0

for fold, (train_index, test_index) in enumerate(skf.split(df['text'], df['label'])):
    print(f'Fold {fold + 1}/{k_folds}')

    validation_losses_per_epoch = []

    f1_scores_per_epoch = []

    train_texts, train_labels = df.loc[train_index, 'text'].tolist(), df.loc[train_index, 'label'].tolist()
    test_texts, test_labels = df.loc[test_index, 'text'].tolist(), df.loc[test_index, 'label'].tolist()

    train_dataset = preprocess_data(train_texts, train_labels)
    test_dataset = preprocess_data(test_texts, test_labels)

    model = AutoModelForSequenceClassification.from_pretrained("indolem/indobert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir=f"./results_fold_{fold}",
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=32,
        evaluation_strategy=IntervalStrategy.EPOCH,
        logging_dir=f"./logs_fold_{fold}",
        logging_steps=100,
        learning_rate=5e-5,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        callbacks=[CustomCallback()],
    )

    def custom_data_collator(features: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = {}
        batch['input_ids'] = torch.stack([feature[0] for feature in features])
        batch['attention_mask'] = torch.stack([feature[1] for feature in features])
        batch['labels'] = torch.tensor([feature[2] for feature in features])
        return batch

    trainer.data_collator = custom_data_collator

    original_evaluate = trainer.evaluate

    def evaluate_with_f1(*args, **kwargs):
        results = original_evaluate(*args, **kwargs)
        predictions = trainer.predict(test_dataset)
        labels = predictions.label_ids
        preds = predictions.predictions.argmax(-1)
        f1 = f1_score(labels, preds)
        f1_scores_per_epoch.append(f1)
        validation_losses.append(results['eval_loss'])
        validation_losses_per_epoch.append(results['eval_loss'])
        return results

    trainer.evaluate = evaluate_with_f1

    original_log = trainer.log

    def log_with_loss(*args, **kwargs):
        original_log(*args, **kwargs)
        if 'loss' in kwargs:
            training_losses.append(kwargs['loss'])

    trainer.log = log_with_loss

    trainer.train()

    predictions = trainer.predict(test_dataset)
    y_true = test_labels
    y_pred = predictions.predictions.argmax(-1)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    metrics['accuracy'].append(accuracy)
    metrics['precision'].append(precision)
    metrics['recall'].append(recall)
    metrics['f1'].append(f1)

    print(f'Fold {fold + 1} Results:')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Ensure the directory exists
    output_dir = '/nas.dbms/fathan/test/multilang-hate-models/IndobertHatespeech/'
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix Fold {fold + 1}')
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_fold_{fold + 1}.png'))

    plt.figure(figsize=(10, 6))
    plt.plot(f1_scores_per_epoch, marker='o', linestyle='-', color='b', label='F1 Score')
    plt.title(f'F1 Score per Epoch Fold {fold + 1}')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'f1_score_per_epoch_fold_{fold + 1}.png'))

    plt.figure(figsize=(10, 6))
    plt.plot(validation_losses_per_epoch, marker='o', linestyle='-', color='g', label='Validation Loss')
    plt.title(f'Validation Loss per Epoch Fold {fold + 1}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'loss_per_epoch_fold_{fold + 1}.png'))

avg_accuracy = np.mean(metrics['accuracy'])
avg_precision = np.mean(metrics['precision'])
avg_recall = np.mean(metrics['recall'])
avg_f1 = np.mean(metrics['f1'])

print("Average Results:")
print(f"Accuracy: {avg_accuracy}")
print(f"Precision: {avg_precision}")
print(f"Recall: {avg_recall}")
print(f"F1 Score: {avg_f1}")
