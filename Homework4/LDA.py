import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


class MyLDA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.predicted_labels: np.ndarray
        self.model: LDA

    def train(self, train_data, train_label):
        self.model = LDA(n_components=self.n_components)
        self.model.fit(train_data, train_label)

    def predict(self, test_data, test_label):
        self.predicted_labels = self.model.predict(test_data)
        correct_count = np.sum(self.predicted_labels == test_label)
        accuracy = correct_count / len(test_label)
        return accuracy


if __name__ == '__main__':
    from preprocess import read_img
    from sklearn.model_selection import train_test_split

    path = 'Grp13Dataset/'
    wide = 70
    height = 80
    images, labels = read_img(path, wide, height)
    train_x, test_x, train_y, test_y = train_test_split(images, labels, test_size=0.3)

    model = MyLDA(10)
    model.train(train_x, train_y)
    acc = model.predict(test_x, test_y)
    print("n_components: " + str(model.n_components) + " accuracy:" + str(acc))