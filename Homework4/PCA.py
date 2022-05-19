import numpy as np
from sklearn.decomposition import PCA


def debug_array(array: np.ndarray):
    print(array.shape)
    print(array)


class MyPCA:
    def __init__(self, ratio=0.99):
        self.ratio = ratio
        self.model: PCA
        self.transformed_train: np.ndarray
        self.train_labels: np.ndarray
        self.n_components: int
        self.predicted_labels: np.ndarray

    def train(self, train_images, train_labels):
        # 求协方差矩阵
        avg_images = np.mean(train_images, axis=0)
        diff_images = train_images - avg_images
        cov_mat = np.cov(diff_images)

        # 计算特征值和特征向量
        eig_val, eig_vec = np.linalg.eig(cov_mat)
        # eig_val_sorted_indices = np.argsort(-eig_val)
        eig_val_sorted = abs(np.sort(-eig_val))
        # eig_vec_sorted = eig_vec[:, eig_val_sorted_indices]

        # 构造特征脸空间
        eig_val_sum = np.sum(eig_val_sorted)
        eig_val_cum = np.cumsum(eig_val_sorted)
        eig_val_cum_percent = eig_val_cum / eig_val_sum
        n_components = np.argmax(eig_val_cum_percent > self.ratio) + 1

        # print("n_components:", n_components)

        _model = PCA(n_components=n_components)
        _transformed_train = _model.fit_transform(train_images)
        self.model = _model
        self.n_components = n_components

        _sort_train = []
        _train_labels = []
        for i in range(1, np.unique(train_labels).size + 1):
            indices = np.where(train_labels == i)
            avg_train_images = np.mean(_transformed_train[indices], axis=0)
            _sort_train.append(avg_train_images)
            _train_labels.append(i)

        self.transformed_train = np.array(_sort_train)
        self.train_labels = np.array(_train_labels)
        return _model

    def predict(self, test_images, test_labels):
        transformed_test = self.model.transform(test_images)

        predicted_labels = []
        for i in range(len(transformed_test)):
            diff = self.transformed_train - transformed_test[i]
            distance = np.sum(np.square(diff), axis=1)
            min_index = np.argmin(distance)
            predicted_labels.append(self.train_labels[min_index])

        # 计算准确率
        correct_count = 0
        for i in range(len(predicted_labels)):
            if predicted_labels[i] == test_labels[i]:
                correct_count += 1
        accuracy = correct_count / len(predicted_labels)
        self.predicted_labels = np.array(predicted_labels)
        # print("accuracy:", accuracy)
        return accuracy
