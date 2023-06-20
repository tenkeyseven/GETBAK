import torch
import numpy as np
from sklearn import metrics
import torch.nn as nn
from material.models.generators import *
import matplotlib.pyplot as plt

# result_file = '/home/nas928/ln/GETBAK/defense_result/strip/0831-linf20/defense_strip.npy'
result_file = '/home/nas928/ln/GETBAK/defense_result/strip/0831-linf25/defense_strip.npy'

_dict = np.load(result_file, allow_pickle=True)

# print(_dict)
# print(type(_dict))

clean_entropy = _dict.item()['clean']
poison_entropy = _dict.item()['poison']

entropy_benigh = clean_entropy
entropy_trojan = poison_entropy

bins = 30
plt.figure(figsize=(10,10))
plt.hist(entropy_benigh, bins, weights=np.ones(len(entropy_benigh)) / len(entropy_benigh), alpha=1, label='without trigger')
plt.hist(entropy_trojan, bins, weights=np.ones(len(entropy_trojan)) / len(entropy_trojan), alpha=0.90, label='with trigger')
plt.legend(loc='upper right', fontsize = 23)
plt.ylabel('Probability (%)', fontsize = 26)
plt.title('normalized entropy', fontsize = 25)
plt.tick_params(labelsize=20)

fig1 = plt.gcf()
plt.show()
# fig1.savefig('EntropyDNNDist_T2.pdf')# save the fig as pdf file
fig1.savefig('/home/nas928/ln/GETBAK/defense_result/strip/results_new.png')# save the fig as pdf file

print('File Saved at : ', result_file)
print('Entropy Clean  Median: ', float(np.median(clean_entropy)))
print('Entropy Poison Median: ', float(np.median(poison_entropy)))

# threshold_low = float(clean_entropy[int(0.05 * len(clean_entropy))])
# threshold_high = float(clean_entropy[int(0.95 * len(clean_entropy))])

# y_true = torch.cat((torch.zeros_like(clean_entropy),
#                     torch.ones_like(poison_entropy)))
# entropy_t = torch.cat((clean_entropy, poison_entropy))
# y_pred = torch.where(((entropy_t < threshold_low).int() + (entropy_t > threshold_high).int()
#                         ).bool(), torch.ones_like(entropy_t), torch.zeros_like(entropy_t))

# print(f'Threshold: ({threshold_low:5.3f}, {threshold_high:5.3f})')
# print("f1_score:", metrics.f1_score(y_true, y_pred))
# print("precision_score:", metrics.precision_score(y_true, y_pred))
# print("recall_score:", metrics.recall_score(y_true, y_pred))
# print("accuracy_score:", metrics.accuracy_score(y_true, y_pred))