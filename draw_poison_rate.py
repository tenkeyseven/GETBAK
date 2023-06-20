import matplotlib.pyplot as plt

index = [1,2,3,4,5,6,7,8,9,10]
acc = [0.9814, 0.9811, 0.9783, 0.9778, 0.9789, 0.9808, 0.9788, 0.9783, 0.9760, 0.9757]
asr = [0.9256, 0.8845, 0.8784, 0.8848, 0.9898, 0.9832, 0.9449, 0.9710, 0.9605, 0.9770]

# with open('/home/nas928/ln/GETBAK/defense_result/fine-pruning/denfense_fine_tune_file.txt', 'r') as f:
#     for line in f.readlines():
#         a = line.split()
#         index.append(float((a[0])))
#         acc.append(float(a[1]))
#         asr.append(float(a[2]))

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
# plt.figure(figsize=(10,5))
# l1=plt.plot(index,asr,'r+-',label='ASR')
# l2=plt.plot(index,acc,'g+-',label='ACC')

# # plt.plot(index,asr,'ro-',index,acc,'g+-')
# # plt.title('Fine-Pruning Result')
# plt.xlabel('Poison Rate (%)')
# plt.ylabel('BA / ASR (%)')
# plt.ylim(0, 1.0)
# plt.legend()
# plt.show()

# plt.savefig('/home/nas928/ln/GETBAK/results/poison_rate_results.png')


# x = np.arange(1, 11, 1)  # x坐标

index = [1,2,3,4,5,6,7,8,9,10]
ba = [0.9814, 0.9811, 0.9783, 0.9778, 0.9789, 0.9808, 0.9788, 0.9783, 0.9760, 0.9757]
asr = [0.9256, 0.8845, 0.8784, 0.8848, 0.9898, 0.9832, 0.9449, 0.9710, 0.9605, 0.9770]

CLBA_ba=[0.9811, 0.9814, 0.9785, 0.9793, 0.9798, 0.9788, 0.9752, 0.9785, 0.9768, 0.9780] 
CLBA_asr=[0.1212, 0.1192, 0.1329, 0.1640, 0.1503, 0.1368, 0.1429, 0.1569, 0.1569, 0.1905]


GRTBA_ba = [0.9808, 0.9815, 0.9811, 0.9814, 0.9788, 0.9824, 0.9819, 0.9770, 0.9740, 0.9735]
GRTBA_asr = [0.2056, 0.1477, 0.4323, 0.4074, 0.4129, 0.4226, 0.4601, 0.3859, 0.4264, 0.4583]

plt.plot(index, asr, lw=1, c='darkred', marker='o', label='ASR(Ours)')  # 绘制y1
plt.plot(index, ba, lw=1, c='darkgreen', marker='o', label='BA(Ours)')  # 绘制y2


plt.plot(index, CLBA_asr, lw=1, c='red', marker='s', label='ASR(CLBA)')  # 绘制y1
plt.plot(index, CLBA_ba, lw=1, c='g', marker='s', label='BA(CLBA)')  # 绘制y2

plt.plot(index, GRTBA_asr, lw=1, c='brown', marker='^', label='ASR(GRTBA)')  # 绘制y1
plt.plot(index, GRTBA_ba, lw=1, c='lime', marker='^', label='BA(GRTBA)')  # 绘制y2

# plt-style 
plt.xticks(index)  # x轴的刻度
# plt.xlim(0, 10)  # x轴坐标范围
plt.ylim(0, 1.0)  # y轴坐标范围
plt.xlabel('Poison Rate (%)')
plt.ylabel('BA / ASR (%)')
plt.legend(bbox_to_anchor=(0.7,0.6,0.3,0.1))  # 图例
plt.savefig('/home/nas928/ln/GETBAK/results/poison_rate_results.png')
plt.show()
