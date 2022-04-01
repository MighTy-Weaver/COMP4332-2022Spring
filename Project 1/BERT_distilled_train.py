import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import DistilBertModel
from transformers import get_scheduler
from transformers.modeling_outputs import TokenClassifierOutput


class CustomModel(torch.nn.Module):
    def __init__(self, num_labels=5, checkpoint=None):
        super(CustomModel, self).__init__()
        self.num_labels = num_labels

        # Load Model with given checkpoint and extract its body
        if checkpoint is None:
            checkpoint = "distilbert-base-uncased"
        self.model = DistilBertModel.from_pretrained(
            checkpoint,
            config=AutoConfig.from_pretrained(checkpoint, output_attentions=True, output_hidden_states=True))
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, num_labels)  # load and initialize weights

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Add custom layers
        sequence_output = self.dropout(outputs[0])  # outputs[0]=last hidden state

        logits = self.classifier(sequence_output[:, 0, :].view(-1, 768))  # calculate losses

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,
                                     attentions=outputs.attentions)


def load_data(split_name='train', columns=None, folder='data'):
    """
        "split_name" may be set as 'train', 'valid' or 'test' to load the corresponding dataset.

        You may also specify the column names to load any columns in the .csv data file.
        Among many, "text" can be used as model input, and "stars" column is the labels (sentiment).
        If you like, you are free to use columns other than "text" for prediction.
    """
    if columns is None:
        columns = ['text', 'stars']
    try:
        print(f"select [{', '.join(columns)}] columns from the {split_name} split")
        df = pd.read_csv(f'{folder}/{split_name}.csv')
        df = df.loc[:, columns]
        print("Success")
        return df
    except:
        print(f"Failed loading specified columns... Returning all columns from the {split_name} split")
        df = pd.read_csv(f'{folder}/{split_name}.csv')
        return df


train_df = load_data('train', columns=['text', 'stars'])
valid_df = load_data('valid', columns=['text', 'stars'])
# the test set labels (the 'stars' column) are not available! So the following code will instead return all columns
test_df = load_data('test', columns=['text', 'stars'])

# Prepare the data.
# As an example, we only use the text data.
x_train = train_df['text']
y_train = train_df['stars']

x_valid = valid_df['text']
y_valid = valid_df['stars']

x_test = test_df['text']

x_train_processed = pd.DataFrame(
    {'text': x_train, 'label': np.array(y_train.to_list()) - 1})
x_valid_processed = pd.DataFrame(
    {'text': x_valid, 'label': np.array(y_valid.to_list()) - 1})

train_dataset = Dataset.from_pandas(x_train_processed)
valid_dataset = Dataset.from_pandas(x_valid_processed)

device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = CustomModel().to(device)


def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)


train_dataset_tokenized = train_dataset.map(tokenize_function, batched=True)
train_dataset_tokenized = train_dataset_tokenized.remove_columns(['text'])
train_dataset_tokenized = train_dataset_tokenized.rename_column(
    "label", "labels")
train_dataset_tokenized.set_format('torch')

valid_dataset_tokenized = valid_dataset.map(tokenize_function, batched=True)
valid_dataset_tokenized = valid_dataset_tokenized.remove_columns(['text'])
valid_dataset_tokenized = valid_dataset_tokenized.rename_column(
    "label", "labels")
valid_dataset_tokenized.set_format('torch')

train_dataloader = DataLoader(train_dataset_tokenized, shuffle=True, batch_size=16)
valid_dataloader = DataLoader(valid_dataset_tokenized, batch_size=16)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 100
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(name='linear', optimizer=optimizer,
                             num_warmup_steps=0, num_training_steps=num_training_steps)

progress_bar = tqdm(range(num_training_steps))

max_val_f1 = 0
max_f1_acc = 0
max_metrics = None

for epoch in range(num_epochs):
    model.train()
    total_acc, total_loss, total_count = 0, 0, 0
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        total_acc += (predictions == batch['labels']).sum().item()
        total_loss += loss.item()
        total_count += batch['labels'].size(0)

        progress_bar.update(1)
        progress_bar.set_postfix({'epoch': epoch,
                                  'loss': total_loss / total_count,
                                  'acc': total_acc / total_count})

    model.eval()
    y_valid_labels, y_pred_labels = [], []
    for batch in valid_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        y_valid_labels.append(batch['labels'].cpu().numpy())
        y_pred_labels.append(predictions)
    y_valid_labels = np.array(y_valid_labels).reshape(-1)
    y_pred_labels = np.array(y_pred_labels).reshape(-1)
    print(classification_report(y_valid_labels, y_pred_labels))
    macro_f1 = classification_report(y_valid_labels, y_pred_labels, output_dict=True)['macro avg']['f1-score']
    print('val accuracy', np.mean(y_valid_labels == y_pred_labels))
    if macro_f1 >= max_val_f1:
        max_val_f1 = macro_f1
        max_f1_acc = np.mean(y_valid_labels == y_pred_labels)
        max_metrics = classification_report(y_valid_labels, y_pred_labels)
        torch.save(model, './BART_val_best.pkl')
    print('\n\n\n---------------------------------\n'
          'MAX F1 {}\tMAX ACC {}\n{}'
          '---------------------------------------\n\n'.format(max_val_f1, max_f1_acc, max_metrics))
