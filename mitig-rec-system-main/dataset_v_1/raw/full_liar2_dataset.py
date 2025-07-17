from dataset_v_1 import load_dataset
import pandas as pd

dataset = load_dataset("chengxuphd/liar2")
train_split = dataset['train']
df_train = pd.DataFrame(train_split)
df_train.to_csv('liar2_train_dataset.csv', index=False)

print(df_train.head())