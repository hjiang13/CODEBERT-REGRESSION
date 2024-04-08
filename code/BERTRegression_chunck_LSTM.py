import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch import nn
import json

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from transformers import LongformerConfig, LongformerModel


trainDataPath = "../dataset/SDC_train_resilience_r.jsonl"
evalDataPath = "../dataset/SDC_test_resilience_r.jsonl"
train_data = pd.DataFrame( {"code": [], "label": []}) 
eval_data = pd.DataFrame( columns=['code', 'label'])

with open(trainDataPath, "r") as data_file:
    i = 0
    for line in data_file:
        line = json.loads(line)
        lineList= [[line["code"], line["label"]]]
        df_line = pd.DataFrame(lineList, columns=['code', 'label'])
        train_data = pd.concat([train_data, df_line])
        i += 1
        #if i > 5:
        #    break

with open(evalDataPath, "r") as data_file:
    i = 0
    for line in data_file:
        line = json.loads(line)
        lineList= [[line["code"], line["label"]]]
        df_line = pd.DataFrame(lineList, columns=['code', 'label'])
        eval_data = pd.concat([eval_data, df_line])
        i += 1
        #if i > 5:
        #    break


#data = pd.read_json("../dataset/SDC_train_resilience_r.jsonl")

# define a datasets
class TextSegmentationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_token_len=512, chunk_size=510):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.tokenizer.encode_plus(text, max_length=self.max_token_len, 
                                             truncation=True, padding='max_length',
                                             add_special_tokens=True, return_tensors="pt")
        input_ids = tokens['input_ids'].squeeze()
        attention_mask = tokens['attention_mask'].squeeze()

        # Segmenting the input_ids & attention_mask
        total_chunks = len(input_ids) // self.chunk_size
        input_ids_chunks = input_ids.unfold(0, self.chunk_size, self.chunk_size)[:total_chunks]
        attention_mask_chunks = attention_mask.unfold(0, self.chunk_size, self.chunk_size)[:total_chunks]

        return {
            'input_ids': input_ids_chunks,
            'attention_mask': attention_mask_chunks,
            'labels': torch.tensor(label, dtype=torch.float)
        }

# Initialize tokenizer
tokenizer = RobertaTokenizer.from_pretrained("neulab/codebert-cpp")


#train_data, eval_data = train_test_split(data, test_size=0.1)
train_dataset = SentimentDataset(train_data['code'].to_numpy(), train_data['label'].to_numpy(), tokenizer)
eval_dataset = SentimentDataset(eval_data['code'].to_numpy(), eval_data['label'].to_numpy(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(eval_dataset, batch_size=1)

# Define a regression model on BERT
class BertRegressor(nn.Module):
    def __init__(self, bert_model="neulab/codebert-cpp", lstm_hidden_size=256, output_size=1):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, lstm_hidden_size, batch_first=True)
        self.regressor = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, input_ids, attention_mask):
        # Flatten input for BERT processing
        batch_size, seq_len, chunk_size = input_ids.size()
        input_ids = input_ids.view(-1, chunk_size)
        attention_mask = attention_mask.view(-1, chunk_size)

        with torch.no_grad():
            bert_output = self.bert(input_ids, attention_mask=attention_mask)

        # Extract [CLS] embeddings
        cls_embeddings = bert_output.last_hidden_state[:, 0, :].view(batch_size, seq_len, -1)

        # LSTM processing
        _, (hidden, _) = self.lstm(cls_embeddings)

        # Regression
        return self.regressor(hidden.squeeze(0))

model = BertRegressor()

# Def optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.MSELoss()

# Train the model
model.train()
for epoch in range(5):  # To be changed
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")

# Evaluation
model.eval()
for batch in val_loader:
    with torch.no_grad():
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        print(f"Predicted label: {outputs.squeeze().item()}, Actual label: {batch['labels'].item()}")