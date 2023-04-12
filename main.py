from datasets import load_dataset
from torch.utils.data import RandomSampler
dataset = load_dataset('multi_nli')




validation_matched = dataset['validation_matched']
validation_mismatched = dataset['validation_mismatched']

print(validation_matched)
print(RandomSampler(validation_mismatched,num_samples=2500))