from datasets import load_dataset
from torch.utils.data import RandomSampler
import torch
import pandas as pd
dataset = load_dataset('multi_nli')




validation_matched = dataset['validation_matched']
validation_mismatched = dataset['validation_mismatched']
torch.manual_seed(42)

validation_matched = validation_matched[list(RandomSampler(validation_matched,num_samples=2500))]
validation_mismatched = validation_mismatched[list(RandomSampler(validation_mismatched,num_samples=2500))]



df = pd.DataFrame()
df['premise'] = validation_matched['premise']
df['hypothesis'] = validation_matched['hypothesis']
df['label'] = validation_matched['label']
df.to_csv('data/validation_matched.csv',index=False)

df = pd.DataFrame()
df['premise'] = validation_mismatched['premise']
df['hypothesis'] = validation_mismatched['hypothesis']
df['label'] = validation_mismatched['label']
df.to_csv('data/validation_mismatched.csv',index=False)