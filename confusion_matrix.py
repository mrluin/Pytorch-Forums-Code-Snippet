import torch

nb_samples = 20
nb_classes = 4
output = torch.randn(nb_samples, nb_classes)
pred = torch.argmax(output, 1)
target = torch.randint(0, nb_classes, (nb_samples,))

conf_matrix = torch.zeros(nb_classes, nb_classes)
for t, p in zip(target, pred):
    conf_matrix[t, p] += 1

print('Confusion matrix\n', conf_matrix)

# print(confusion_matrix.diag()/confusion_matrix.sum(1)) for per-class accuracy


# matrix.diag() 对角元
TP = conf_matrix.diag()
#print(TP)
print("---------------------------------------")
for c in range(nb_classes):
    # pay attention: index uint8 is different with long
    idx = torch.ones(nb_classes).byte()
    idx[c] = 0
    # all non-class samples classified as non-class
    #print(idx.nonzero()[:,None])
    '''
    print('idx-nonzero', idx.nonzero())
    print(conf_matrix[idx.nonzero()])
    print(conf_matrix[idx.nonzero(), idx.nonzero()])
    print(conf_matrix[idx.nonzero()[:,None]])
    print(conf_matrix[idx.nonzero()[:, None], idx.nonzero()])
    print("----------------------------------------")
    '''
    TN = conf_matrix[
        idx.nonzero()[:, None], idx.nonzero()].sum()  # conf_matrix[idx[:, None], idx].sum() - conf_matrix[idx, c].sum()
    # all non-class samples classified as class
    FP = conf_matrix[idx, c].sum()
    # all class samples not classified as class
    FN = conf_matrix[c, idx].sum()

    print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(
        c, TP[c], TN, FP, FN))



'''
# index type byte or long is different completely
index1 = torch.tensor([0,1,0,1], dtype=torch.uint8)
index2 = torch.tensor([0,1,0,1], dtype=torch.long)
data = torch.tensor([1,2,3,4])

print(data[index1]) # output 2,4
print(data[index2]) # output 1 2 1 2
'''
'''
# In numpy, selection by [:,None]的问题 增加维度
'''
batch_size = 32
nb_classes = 4

output = torch.randn(batch_size, nb_classes)
target = torch.randint(0, nb_classes, (batch_size,))

def confusion_matrix(preds, labels):

    preds = torch.argmax(preds, 1)
    conf_matrix = torch.zeros(nb_classes, nb_classes)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1

    print(conf_matrix)
    TP = conf_matrix.diag()
    for c in range(nb_classes):
        idx = torch.ones(nb_classes).byte()
        idx[c] = 0
        TN = conf_matrix[idx.nonzero()[:,None], idx.nonzero()].sum()
        FP = conf_matrix[c, idx].sum()
        FN = conf_matrix[idx, c].sum()

        sensitivity = (TP[c] / (TP[c]+FN))
        specificity = (TN / (TN+FP))

        print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(
            c, TP[c], TN, FP, FN))
        print('Sensitivity = {}'.format(sensitivity))
        print('Specificity = {}'.format(specificity))

confusion_matrix(output, target)