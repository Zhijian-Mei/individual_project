from datasets import load_dataset

dataset = load_dataset('multi_nli')

validation_matched = dataset['validation_matched']
validation_mismatched = dataset['validation_mismatched']

print(validation_matched)