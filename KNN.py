import numpy as np


class KNearestNeighbor:
    def __init__(self, K, X, Y, weighted=False):
        self.K = K  # K
        self.X = X  # features
        self.Y = Y  # target
        self.weighted = weighted  # weighted majority vote 사용 여부(default는 False)

    # euclidian distance 계산 method
    def calculate_euclidian_distance(self, x, y):
        # overflow 방지를 위한 converting
        x = np.uint32(x)
        y = np.uint32(y)
        return np.sqrt(np.sum((x - y) ** 2))

    # k개의 인접한 개체를 구하여 반환하는 method
    def obtain_k_nearest_neighbors(self, y):
        # calculate_euclidian_distance method를 사용하여 입력받은 y값에 대해 (index, euclidian distance) 형식의 tuple로 이루어진 list 생성
        euclidian_distances = [(i, self.calculate_euclidian_distance(x, y)) for i, x in enumerate(self.X)]
        euclidian_distances.sort(key=lambda tup: tup[1])  # euclidian distance를 기준으로 sort
        return euclidian_distances[:self.K]  # 상위 k개의 list 반환

    # majority vote method
    def obtain_majority_vote(self, nearest_neighbors):
        # 입력 nearest_neighbors는 k개 nearest neighbor의 class들로 이루어진 array
        votes = {}  # class가 key, vote 수가 value가 되는 dictionary
        # vote 진행
        for nearest_neighbor in nearest_neighbors:
            if nearest_neighbor in votes:
                votes[nearest_neighbor] += 1
            else:
                votes[nearest_neighbor] = 1
        for voted_class, vote_count in votes.items():
            if vote_count == max(votes.values()):
                return voted_class  # 가장 vote 수가 높은 class를 반환
        raise Exception("most voted class가 존재하지 않습니다")

    # weighted majority vote method
    def obtain_weighted_majority_vote(self, nearest_neighbors):
        # 입력 nearest_neighbors는 k개 nearest neighbor의 (class, euclidian distance) tuple 들로 이루어진 list
        votes = {}  # class가 key, vote 수가 value가 되는 dictionary
        # vote 진행
        for nearest_neighbor in nearest_neighbors:
            if nearest_neighbor[0] in votes:
                votes[nearest_neighbor[0]] += (1 + 1 / nearest_neighbor[1])  # euclidian distance에 따른 가중치도 vote에 반영
            else:
                votes[nearest_neighbor[0]] = (1 + 1 / nearest_neighbor[1])
        for voted_class, vote_count in votes.items():
            if vote_count == max(votes.values()):
                return voted_class  # 가장 vote 수가 높은 class를 반환
        raise Exception("most voted class가 존재하지 않습니다")

    # classification method
    def classify(self, data):
        result_class = []  # 결과 값을 담을 빈 list
        for i, datum in enumerate(data):
            k_nearest_neighbors = self.obtain_k_nearest_neighbors(datum)  # k개 nearest neighbors 계산
            if self.weighted:
                # weighted가 true일 경우 (class, euclidian distance) tuple로 이루어진 list 생성
                k_nearest_neighbors_tuple = [(item, tup[1]) for i, item in enumerate(self.Y) for tup in
                                             k_nearest_neighbors if tup[0] == i]
                result_class.append(self.obtain_weighted_majority_vote(k_nearest_neighbors_tuple))
            else:
                # weighted가 false일 경우 weighted가 true일 때와 달리 euclidian distance는 제외하고 k개 nearest neighbors의 class들만 넘겨줌
                result_class.append(self.obtain_majority_vote(
                    self.Y[[euclidian_distance[0] for euclidian_distance in k_nearest_neighbors]]))
        return result_class
