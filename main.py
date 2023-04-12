from datasets import load_dataset
from torch.utils.data import RandomSampler
import torch
dataset = load_dataset('multi_nli')




validation_matched = dataset['validation_matched']
validation_mismatched = dataset['validation_mismatched']
torch.manual_seed(42)

validation_matched = validation_matched[list(RandomSampler(validation_matched,num_samples=2500))]
validation_mismatched = validation_mismatched[list(RandomSampler(validation_mismatched,num_samples=2500))]

print(validation_matched[0])