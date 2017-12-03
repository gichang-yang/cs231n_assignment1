import cs231n.classifiers.k_nearest_neighbor as knn
import cs231n.data_utils as du
import numpy as np
Xtr, Ytr, Xte, Yte = du.load_CIFAR10('./datasets/cifar-10-batches-py')
Xtr = np.array(Xtr)
Ytr = np.array(Ytr)
Xte = np.array(Xte)
Yte = np.array(Yte)
print(Xtr)
Xtr_rows = np.reshape(Xtr,[-1,32*32*3])
print(Xtr_rows.shape)
Xte_rows = np.reshape(Xte,[-1,32*32*3])
print(Xte_rows.shape)

nn = knn.KNearestNeighbor()
nn.train(Xtr_rows,Ytr)
print('train end')
Yte_predict = nn.predict(Xte_rows)

print('accuracy: %f' % (np.mean(Yte_predict == Yte)))
print('debug')