import numpy as np
from sklearn.datasets import load_iris
from KNN import KNearestNeighbor

# train data와 test data를 나누는 함수
def split_train_test(data, target):
    # split한 데이터를 넣어주기 위해서 array 생성
    # 새로 생성된 data array는 인덱스 0의 값이 [0., 0., 0., 0.]이, target array는 인덱스 0의 값이 0.0이 됨
    # data를 10등분 -> data 개수가 150개 이므로 15개씩 나눠짐
    train_data = test_data = np.zeros((1, 4)) 
    splitted_data = np.split(data, 10) 
    train_target = test_target = np.zeros(1)
    splitted_target = np.split(target, 10)

    # 위에서 새로 생성한 array들 중 train용에는 14번째 요소까지 넣고 test용에는 마지막 15번째 요소만 append
    for splitted_datum in splitted_data:
        train_data = np.append(train_data, splitted_datum[:14], axis=0)
        test_data = np.append(test_data, splitted_datum[-1:], axis=0)
    for splitted_datum in splitted_target:
        train_target = np.append(train_target, splitted_datum[:14], axis=0)
        test_target = np.append(test_target, splitted_datum[-1:], axis=0)

    # 아까 새로 생성했던 array들의 0번째 인덱스는 0이거나 0으로된 array이므로 해당 값은 제외하고 반환
    return train_data[1:], test_data[1:], train_target[1:], test_target[1:]


if __name__ == "__main__":
    iris = load_iris()

    label_dict = {0: "setosa", 1: "versicolor", 2: "virginica"} # 각 class에 대응되는 숫자를 target 이름과 매치시킨 dictionary
    train_data, test_data, train_label, test_label = split_train_test(iris['data'], iris['target'])

    k_list = [3, 5, 9] # iteration에 사용할 각 k값들의 list

    # majority vote를 사용한 KNN
    print("==========unweighted==========")
    for k in k_list:
        unweighted_accuracy = 0
        knn = KNearestNeighbor(k, train_data, train_label)
        print(f"k={k}")
        predicted_result = knn.classify(test_data)
        for i, e in enumerate(predicted_result):
            if label_dict[e] == label_dict[test_label[i]]:
                unweighted_accuracy += 1
            print(f"Test Data Index: {i}\t Computed class: {label_dict[e]}\t True class: {label_dict[test_label[i]]}")
        unweighted_accuracy = (unweighted_accuracy / 10) * 100
        print(f"Accuracy: {unweighted_accuracy}%\n")

    # weighted majority vote를 사용한 KNN
    print("==========weighted==========")
    for k in k_list:
        weighted_accuracy = 0
        knn = KNearestNeighbor(k, train_data, train_label, weighted=True)
        print(f"k={k}")
        predicted_result = knn.classify(test_data)
        for i, e in enumerate(predicted_result):
            if label_dict[e] == label_dict[test_label[i]]:
                weighted_accuracy += 1
            print(f"Test Data Index: {i}\t Computed class: {label_dict[e]}\t True class: {label_dict[test_label[i]]}")
        weighted_accuracy = (weighted_accuracy / 10) * 100
        print(f"Accuracy: {weighted_accuracy}%\n")
