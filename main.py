from datasets import load_dataset

dataset = load_dataset('multi_nli')

def f(x):
    return {'premise':x['premise'],'hypothesis':x['hypothesis']}


validation_matched = dataset['validation_matched'].map(f,batched=True)
validation_mismatched = dataset['validation_mismatched'].map(lambda x:{'premise':x['premise'],'hypothesis':x['hypothesis']})

print(validation_matched)