import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from decimal import getcontext, Decimal

getcontext().prec = 28


def create_X_matrix(dataset,vocab,mydict):
    X_structure = [len(np.unique(dataset[:, 0])), len(vocab)]
    X = np.zeros(X_structure)
    for row in dataset:
        doc_id = row[0]
        word_id = row[1]
        freq = row[2]
        if word_id in mydict:
            X[doc_id - 1, word_id - 1] = freq
    return X


def create_Dictionary(dataset):
    dict = {}
    for row in dataset:
        dict[row[1]] = dict.get(row[1],0) + row[2]
    dict = [k for k in sorted(dict, key=dict.get, reverse=True)]
    return dict


def create_y(label,classes):
    y_structure = [len(label),len(classes)]
    y = np.zeros(y_structure)
    i = 0
    for row in label:
        y[i,row-1] = 1
        i += 1
    return y


def create_theta(vocab,classes,X,y):
    theta_structure = [len(vocab),len(classes)]
    theta = np.zeros(theta_structure)
    for j in range(len(vocab)):
        for k in range(len(classes)):
            sum_y = np.count_nonzero(y[:,k])
            num = np.sum(np.logical_and(X[:,j],y[:,k]))
            theta[j][k] = (num + 1)/(sum_y + 2)
    return theta


def predict(X_test,theta,y,classes):
    ypred = []
    i=1
    for x in X_test:
        print("predicted ",i)
        i += 1
        words = np.where(x > 0.0)
        words_x = np.where(x == 0.0)
        max = 0
        expec = -1
        for k in range(len(classes)):
            p = theta[words,k]
            q = theta[words_x,k]
            prod = Decimal(1)
            for num in p[0]:
                prod = Decimal(prod) * Decimal(num)
            for num in q[0]:
                prod = Decimal(prod) * Decimal(Decimal(1) - Decimal(num))
            prob = Decimal(prod * np.count_nonzero(y[:,k]))
            if prob > max:
                max = prob
                expec = k + 1
        ypred.append(expec)
    return ypred


def main():
    dataset = np.array(pd.read_csv("train.data",header=None,delim_whitespace=True))
    dataset_test = np.array(pd.read_csv("test.data",header=None,delim_whitespace=True))
    vocab = np.array(pd.read_csv("vocabulary.txt",header=None))
    classes = np.array(pd.read_csv("train.map",header=None,delim_whitespace=True))
    label = np.array(pd.read_csv("train.label",header=None))
    label_y = np.array(pd.read_csv("test.label",header=None))
    mydict = create_Dictionary(dataset)
    v = [100, 500, 1000, 2500, 5000, 7500, 10000, 12500, 25000, 50000, len(mydict)]
    accuracy_list = []
    for value in v:
        print("Executing model for vocab size:",value)
        X = create_X_matrix(dataset,vocab,mydict[0:value])
        X_test = create_X_matrix(dataset_test,vocab,mydict[0:value])
        y = create_y(label,classes)
        print("genetaing theta")
        theta = create_theta(vocab,classes,X,y)
        print("predicting")
        ypred = predict(X_test,theta,y,classes)
        from sklearn.metrics import accuracy_score,precision_score,recall_score
        accuracy = accuracy_score(label_y,ypred)
        accuracy_list.append(accuracy)
        print("accuracy is :", accuracy)
        if value == len(mydict):
            labels, indexes = np.unique(label_y, return_index=True)
            indexes.add(len(label_y))
            class_precision = []
            class_recall = []
            for i in range(len(labels)):
                precision = precision_score(label_y[i:i+1],ypred[i:i+1])
                class_precision.append(precision)
                recall =  recall_score(label_y[i:i+1],ypred[i:i+1])
                class_recall.append(recall)
            plt.bar(labels, class_precision)
            plt.title('Precision for all classes')
            plt.xlabel('Class Labels')
            plt.ylabel('Precision')
            plt.show()
            plt.bar(labels, class_recall)
            plt.title('Recall for all classes')
            plt.xlabel('Class Labels')
            plt.ylabel('Recall')
            plt.show()
    plt.plot(v, accuracy_list)
    plt.title('Accuracy vs Vocabulary Size')
    plt.xlabel('Vocabulary Size')
    plt.ylabel('Accuracy')
    plt.show()


if __name__ == '__main__':
    main()
