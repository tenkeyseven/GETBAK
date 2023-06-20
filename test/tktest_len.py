poisoned_train_set = [1,2,3]
test_set = [1,2]

dataset_sizes = {x:len(poisoned_train_set) if x== 'train' else len(test_set) for x in ['train', 'val']}

print(dataset_sizes)