import matplotlib.pyplot as plt

index = []
acc = []
asr = []

with open('/home/nas928/ln/GETBAK/defense_result/fine-pruning/denfense_fine_tune_file.txt', 'r') as f:
    for line in f.readlines():
        a = line.split()
        index.append(float((a[0])))
        acc.append(float(a[1]))
        asr.append(float(a[2]))

# print(asr)


import numpy as np
import matplotlib.pyplot as plt  
# x1=[20,33,51,79,101,121,132,145,162,182,203,219,232,243,256,270,287,310,325]
# y1=[49,48,48,48,48,87,106,123,155,191,233,261,278,284,297,307,341,319,341]
# x2=[31,52,73,92,101,112,126,140,153,175,186,196,215,230,240,270,288,300]
# y2=[48,48,48,48,49,89,162,237,302,378,443,472,522,597,628,661,690,702]
# x3=[30,50,70,90,105,114,128,137,147,159,170,180,190,200,210,230,243,259,284,297,311]
# y3=[48,48,48,48,66,173,351,472,586,712,804,899,994,1094,1198,1360,1458,1578,1734,1797,1892]
# x=np.arange(20,350)

# index = range(512)
# print(index)
plt.figure(figsize=(10,5))
l1=plt.plot(index,asr,'r+-',label='ASR')
l2=plt.plot(index,acc,'g+-',label='ACC')

# plt.plot(index,asr,'ro-',index,acc,'g+-')
plt.title('Fine-Pruning Result')
plt.xlabel('Pruning Number')
plt.ylabel('Rate')
plt.legend()
plt.show()

plt.savefig('/home/nas928/ln/GETBAK/defense_result/fine-pruning/fine_tune_result.png')