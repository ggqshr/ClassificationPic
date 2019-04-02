from util import PairDataSet

data = PairDataSet(r"D:\Project\Pytorch_Project\pytorch_test\pytorchtest\14classes")
count = 0
sum = 0
for d, l in data:
    count += l
    sum += 1
print(count)
print(sum)
