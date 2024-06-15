import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataframe
# Assuming the dataframe is loaded into `df`
data1 = pd.read_csv('/nas.dbms/fathan/test/processed_train.csv')
print(data1.info())
df = data1.iloc[:5596]  # Adjust the path to your data
tf = data1.iloc[5596:]

# Ensure all label columns contain only integers
label_columns = ['pornografi', 'sara', 'radikalisme', 'pencemaran_nama_baik']
df[label_columns] = df[label_columns].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
tf[label_columns] = tf[label_columns].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

# print(df.sample(5))
# print(tf.sample(5))

# Define the dataset class
class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx]['processed_text']
        labels = self.dataframe.iloc[idx][label_columns].values.astype(int)
        labels = torch.tensor(labels, dtype=torch.float32)
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }

# Load the IndoBERT tokenizer
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')

# Create the dataset and dataloader
max_len = 128
dataset = TextDataset(df, tokenizer, max_len)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create the testset and testloader
max_len = 128
testset = TextDataset(tf, tokenizer, max_len)
testloader = DataLoader(testset, batch_size=32, shuffle=True)

# Define the model
class TextClassifier(nn.Module):
    def __init__(self, n_classes):
        super(TextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('indobenchmark/indobert-base-p1')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[1]  # Extract the pooled output
        output = self.drop(pooled_output)
        return self.out(output)

model = TextClassifier(n_classes=4)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.BCEWithLogitsLoss()

# Function to save the confusion matrix
def save_confusion_matrix(y_true, y_pred, class_names, output_file):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    plt.figure(figsize=(10,7))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(output_file)
    plt.close()

# Function to plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    plt.figure(figsize=(10,7))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.show()

# Evaluation function
def eval_model(model, data_loader, loss_fn, device, n_examples, class_names):
    model = model.eval()
    losses = []
    preds = []
    true_labels = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = loss_fn(outputs, labels)
            losses.append(loss.item())

            preds.append(outputs.cpu())
            true_labels.append(labels.cpu())

    preds = torch.cat(preds)
    true_labels = torch.cat(true_labels)

    # Convert logits to probabilities
    preds = torch.sigmoid(preds)
    
    # Apply threshold to get binary predictions
    preds = (preds > 0.5).float()

    accuracy = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average='weighted')
    precision = precision_score(true_labels, preds, average='weighted')
    recall = recall_score(true_labels, preds, average='weighted')

    # Plot confusion matrix for each class
    for i, class_name in enumerate(class_names):
        plot_confusion_matrix(true_labels[:, i], preds[:, i], [f'Not {class_name}', class_name], title=f'Confusion Matrix for {class_name}')

    return np.mean(losses), accuracy, f1, precision, recall

# Training function
def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = loss_fn(outputs, labels)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(losses)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

EPOCHS = 50
class_names = ['pornografi', 'sara', 'radikalisme', 'pencemaran_nama_baik']
save_f1 = []
save_tloss = []
save_vloss = []
for epoch in range(EPOCHS):
    train_loss = train_epoch(
        model,
        dataloader,
        loss_fn,
        optimizer,
        device,
        len(dataset)
    )
    
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print(f'Train loss: {train_loss}')
    
    val_loss, val_accuracy, val_f1, val_precision, val_recall = eval_model(
        model,
        testloader,
        loss_fn,
        device,
        len(testset),
        class_names
    )
    
    print(f'validation loss: {val_loss}')
    print(f'accuracy: {val_accuracy}')
    print(f'F1 score: {val_f1}')
    print(f'precision: {val_precision}')
    print(f'recall: {val_recall}')

    save_f1.append(val_f1)
    save_tloss.append(train_loss)
    save_vloss.append(val_loss)

plt.figure(figsize=(10, 6))
plt.plot(save_f1, marker='o', linestyle='-', color='b', label='F1 Score')
plt.title(f'F1 Score per Epoch')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.grid(True)
plt.legend()
plt.savefig(f'f1_score_per_epoch.png')

plt.figure(figsize=(10, 6))
plt.plot(save_tloss, marker='o', linestyle='-', color='b', label='Training Loss')
plt.plot(save_vloss, marker='o', linestyle='-', color='g', label='Validation Loss')
plt.title(f'Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.savefig(f'loss_per_epoch.png')