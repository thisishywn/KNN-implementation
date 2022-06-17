import os
import sys
import numpy as np
from dataset.mnist import load_mnist
from KNN import KNearestNeighbor

sys.path.append(os.pardir)


def load_mnist_with_hand_crafted_feature(input_size, output_size):
    (train_data_for_hcf, _), (test_data_for_hcf, _) = load_mnist(flatten=False, normalize=False)

    # load_mnist에서 flatten=False 였으므로 shape는 각각 (60000, 1, 28, 28), (10000, 1, 28, 28)
    # 따라서 먼저 (60000, 28, 28), (10000, 28, 28)로 reshape
    temp_train = train_data_for_hcf.reshape(train_data_for_hcf.shape[0], input_size, input_size)
    temp_test = test_data_for_hcf.reshape(test_data_for_hcf.shape[0], input_size, input_size)

    # input_size 와 output_size를 나눈 값으로 배열을 분할한 뒤에 mean값을 이용해
    # 배열을 (60000, output_size, output_size), (10000, output_size, output_size)로 reshape
    divide_size = input_size // output_size
    temp_train = temp_train.reshape((temp_train.shape[0], output_size, divide_size, output_size, divide_size)).mean(
        4).mean(2)
    temp_test = temp_test.reshape((temp_test.shape[0], output_size, divide_size, output_size, divide_size)).mean(
        4).mean(2)

    # (60000, output_size, output_size), (10000, output_size, output_size)로 reshape 된 배열을 flatten
    train_data_flattened = temp_train.reshape((temp_train.shape[0], output_size * output_size))
    test_data_flattened = temp_test.reshape((temp_test.shape[0], output_size * output_size))

    return train_data_flattened, test_data_flattened


if __name__ == "__main__":
    (train_data, train_label), (test_data, test_label) = load_mnist(flatten=True, normalize=False)

    label_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # test 데이터로부터 무작위 1000개 데이터 뽑음
    size = 1000
    sample = np.random.randint(0, test_label.shape[0], size)

    print("========== 784(28*28) ==========")
    # inference
    accuracy = 0
    # k = 9, weighted majority vote 이용
    knn = KNearestNeighbor(9, train_data, train_label, weighted=True)
    predicted_result = knn.classify(test_data[sample])
    for i, e in enumerate(predicted_result):
        if label_name[e] == label_name[test_label[sample[i]]]:
            accuracy += 1
        print(f"{sample[i]}th Data\t Computed class: {label_name[e]}\t True class: {label_name[test_label[sample[i]]]}")
    accuracy = (accuracy / len(sample)) * 100
    print(f"Accuracy = {accuracy}%\n")

    train_data_hcf, test_data_hcf = load_mnist_with_hand_crafted_feature(28, 14)

    print("========== 196(14*14) ==========")
    # inference - handcrafted data
    hcf_accuracy = 0
    knn_hcf = KNearestNeighbor(9, train_data_hcf, train_label, weighted=True)
    predicted_result_hcf = knn_hcf.classify(test_data_hcf[sample])
    for i, e in enumerate(predicted_result_hcf):
        if label_name[e] == label_name[test_label[sample[i]]]:
            hcf_accuracy += 1
        print(
            f"{sample[i]}th Data \t Computed class: {label_name[e]}\t True class: {label_name[test_label[sample[i]]]}")
    hcf_accuracy = (hcf_accuracy / len(sample)) * 100
    print(f"Accuracy = {hcf_accuracy}%\n")
