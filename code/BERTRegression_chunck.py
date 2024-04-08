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


trainDataPath = "../dataset/benign_train_resilience_r.jsonl"
evalDataPath = "../dataset/benign_test_resilience_r.jsonl"
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
        if i > 5:
            break

with open(evalDataPath, "r") as data_file:
    i = 0
    for line in data_file:
        line = json.loads(line)
        lineList= [[line["code"], line["label"]]]
        df_line = pd.DataFrame(lineList, columns=['code', 'label'])
        eval_data = pd.concat([eval_data, df_line])
        i += 1
        if i > 5:
            break


#data = pd.read_json("../dataset/SDC_train_resilience_r.jsonl")

# define a datasets
class SentimentDataset(Dataset):
    def __init__(self, codes, labels, tokenizer, max_len=512):
        self.codes = codes
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.codes)
    
    def __getitem__(self, item):
        code = str(self.codes[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            code,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'code': code,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
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
    def __init__(self):
        super(BertRegressor, self).__init__()
        self.bert = BertModel.from_pretrained('neulab/codebert-cpp', num_labels=1)
        self.regressor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask):
        # To check if the size is larger than max_length
        if input_ids.size(1) > 512:  # 假设batch_size在第一维
            all_embeddings = []
            step_size = 512
            for i in range(0, input_ids.size(1), step_size):
                chunk_input_ids = input_ids[:, i:i+step_size]
                chunk_attention_mask = attention_mask[:, i:i+step_size]
                chunk_outputs = self.bert(input_ids=chunk_input_ids, attention_mask=chunk_attention_mask)
                chunk_embeddings = chunk_outputs.pooler_output
                all_embeddings.append(chunk_embeddings)
            
            # 计算所有chunks embeddings的平均值
            pooled_output = torch.mean(torch.stack(all_embeddings, dim=0), dim=0)
        else:        
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
        return self.regressor(pooled_output)

model = BertRegressor()

# Def optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.MSELoss()

# Train the model
model.train()
for epoch in range(1):  # To be changed
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
