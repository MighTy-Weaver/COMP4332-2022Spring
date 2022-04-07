import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import get_scheduler


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
# x_valid_processed.to_csv('data_processed/train.csv', index=None)
# x_valid_processed .to_csv('data_processed/valid.csv', index=None)
train_dataset = Dataset.from_pandas(x_train_processed)
valid_dataset = Dataset.from_pandas(x_valid_processed)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)


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

train_dataloader = DataLoader(train_dataset_tokenized, shuffle=True, batch_size=10)
valid_dataloader = DataLoader(valid_dataset_tokenized, batch_size=10)

x_test_processed = pd.DataFrame({'text': x_test})
test_dataset = Dataset.from_pandas(x_test_processed)
test_dataset_tokenized = test_dataset.map(tokenize_function, batched=True)
test_dataset_tokenized = test_dataset_tokenized.remove_columns(['text'])
test_dataset_tokenized.set_format('torch')
test_dataloader = DataLoader(test_dataset_tokenized, shuffle=False, batch_size=10)

optimizer = AdamW(model.parameters(), lr=5e-5)

device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
print(device)
# device = torch.device('cpu')
model.to(device)

num_epochs = 100
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(name='linear', optimizer=optimizer,
                             num_warmup_steps=0, num_training_steps=num_training_steps)

progress_bar = tqdm(range(num_training_steps))

max_val_f1 = 0
max_f1_acc = 0
max_metrics = None

record = {'trn_loss': [], 'trn_acc': [], 'val_loss': [], 'val_acc': []}

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

    record['trn_loss'].append(total_loss / total_count)
    record['trn_acc'].append(total_acc / total_count)

    model.eval()
    y_valid_labels, y_pred_labels = [], []
    val_loss = 0
    for batch in valid_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        y_valid_labels.extend(list(batch['labels'].cpu().numpy()))
        y_pred_labels.extend(list(predictions))
        val_loss += outputs.loss.item()
    y_valid_labels = np.array(y_valid_labels).reshape(-1)
    y_pred_labels = np.array(y_pred_labels).reshape(-1)
    print(classification_report(y_valid_labels, y_pred_labels))
    macro_f1 = classification_report(y_valid_labels, y_pred_labels, output_dict=True)['macro avg']['f1-score']
    print('val accuracy', np.mean(y_valid_labels == y_pred_labels))

    record['val_loss'].append(val_loss / len(valid_dataloader))
    record['val_acc'].append(np.mean(y_valid_labels == y_pred_labels))
    np.save('./BERT_recording.npy', record)
    if macro_f1 >= max_val_f1:
        max_val_f1 = macro_f1
        max_f1_acc = np.mean(y_valid_labels == y_pred_labels)
        max_metrics = classification_report(y_valid_labels, y_pred_labels)

        y_test_pred = []
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            y_test_pred.extend(list(predictions))
        pred_df = pd.DataFrame({'review_id': test_df['review_id'], 'stars': y_test_pred, 'text': test_df['text']})
        pred_df['stars'] = pred_df['stars'] + 1
        pred_df.to_csv('./BERT_val_best_pred.csv', index=False)
        torch.save(model, './BERT_val_best.pkl')
    print('\n\n\n---------------------------------\n'
          'MAX F1 {}\tMAX ACC {}\n{}'
          '---------------------------------------\n\n'.format(max_val_f1, max_f1_acc, max_metrics))
