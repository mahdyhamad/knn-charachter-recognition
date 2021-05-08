DEBUG = False

if DEBUG:
    # This code only exists to help us visually inspect the images.
    # It's in an `if DEBUG:` block to illustrate that we don't need it for our code to work.
    from PIL import Image
    import numpy as np

    def read_image(path):
        return np.asarray(Image.open(path).convert('L'))

    def write_image(image, path):
        img = Image.fromarray(np.array(image), 'L')
        img.save(path)


DATA_DIR = 'data/'
TEST_DIR = 'test/'
DATASET = 'mnist'  # `'mnist'`
TEST_DATA_FILENAME = DATA_DIR + 't10k-images.idx3-ubyte'
TEST_LABELS_FILENAME = DATA_DIR + 't10k-labels.idx1-ubyte'
TRAIN_DATA_FILENAME = DATA_DIR + 'train-images.idx3-ubyte'
TRAIN_LABELS_FILENAME = DATA_DIR + 'train-labels.idx1-ubyte'


def bytes_to_int(byte_data):
    return int.from_bytes(byte_data, 'big')


def read_images(filename, n_max_images=None):
    """
    Read the MNIST dataset, the dataset is in binary formate.
    Purpose: Read both testing and training data (images).
    """
    images = []
    with open(filename, 'rb') as f:
        _ = f.read(
            4
        )  # the first 4 bytes (32-bits) are a magic number, we will ignore it

        n_images = bytes_to_int(
            f.read(4)
        )  # the second 4 bytes (32-bits) are the number of images we will have in this file

        if n_max_images:
            n_images = n_max_images

        n_rows = bytes_to_int(
            f.read(4))  # the third 4 bytes (32-bits) are the number of rows
        n_columns = bytes_to_int(f.read(
            4))  # the fourth 4 bytes (32-bits) are the number of columns

        # the rest of the data will be pixels, each pixel is 1 byte (8-bits)

        for image_idx in range(n_images):
            image = []
            for row_idx in range(n_rows):
                row = []
                for col_idx in range(n_columns):
                    pixel = f.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)

    return images


def read_labels(filename, n_max_labels=None):
    """
    Read MNIST labels, the labels will be one dimentional.
    """
    labels = []
    with open(filename, 'rb') as f:
        _ = f.read(4)  # magic number
        n_labels = bytes_to_int(f.read(4))  # the numbers of labels, 4-bytes

        if n_max_labels:
            n_labels = n_max_labels

        for label_idx in range(n_labels):
            label = bytes_to_int(
                f.read(1))  # read the label, 1-byte. Which will be from 0 - 9
            labels.append(label)

    return labels


def flatten_list(l):
    return [pixel for sublist in l for pixel in sublist]


def extract_features(X):
    return [flatten_list(sample) for sample in X]


def dist(x, y):
    """
    Returns the Euclidean distance between vectors `x` and `y`.
    it just calculates the distance between two vectors. What that means?
    When we get the distance between two vectors, we can tell how similar thay are. Less distnce -> more similar.
    """
    return sum([(bytes_to_int(x_i) - bytes_to_int(y_i))**2
                for x_i, y_i in zip(x, y)])**(0.5)


def get_training_distances_for_test_sample(X_train, test_sample):
    """
    returns the distances between a test_sample and all training_samples.
    so we will have a list of distances [215.515, 545.615, ...] each number is the distance between the test sample and a training_sample.
    """
    return [dist(train_sample, test_sample) for train_sample in X_train]


def get_most_frequent_element(l):
    return max(l, key=l.count)


def knn(X_train, y_train, X_test, k=3):
    """
    K nearest neighbor implementation.
    Args:
    X_train: training data set
    y_train: training labels
    X_test: testing data set
    """
    # the labels of our algorithm predictions for every image in X_test.
    y_pred = []
    """
    ex:
    X_test = [img1, img2, ...] img = [ [..row1], [..row2], ... ]
    y_pred = [1, 3, 9, ...]
    """

    # iterate over every test samples, so we can predict the corresponding digit.
    for test_sample_idx, test_sample in enumerate(X_test):
        print(test_sample_idx, end=' ',
              flush=True)  # print the index of the image

        training_distances = get_training_distances_for_test_sample(
            X_train, test_sample)

        # sort distances in ascending order
        sorted_distance_indices = [
            pair[0] for pair in sorted(enumerate(training_distances),
                                       key=lambda x: x[1])
        ]

        # get the first kth labels from the training_data which have the least difference in distance
        candidates = [y_train[idx] for idx in sorted_distance_indices[:k]]

        # findes the most frequent label
        top_candidate = get_most_frequent_element(candidates)

        y_pred.append(top_candidate)
    return y_pred


def main():
    n_train = 10
    n_test = 1
    k = 7
    print(f'Dataset: {DATASET}')
    print(f'n_train: {n_train}')
    print(f'n_test: {n_test}')
    print(f'k: {k}')

    X_train = read_images(TRAIN_DATA_FILENAME, n_train)
    y_train = read_labels(TRAIN_LABELS_FILENAME, n_train)
    X_test = read_images(TEST_DATA_FILENAME, n_test)
    y_test = read_labels(TEST_LABELS_FILENAME, n_test)

    if DEBUG:
        # Write some images out just so we can see them visually.
        for idx, test_sample in enumerate(X_test):
            write_image(test_sample, f'{TEST_DIR}{idx}.png')
        # Load in the `our_test.png` we drew ourselves!
        X_test = [read_image(f'{DATA_DIR}our_test.png')]
        y_test = [3]

    X_train = extract_features(X_train)
    X_test = extract_features(X_test)

    y_pred = knn(X_train, y_train, X_test, k)

    accuracy = sum([
        int(y_pred_i == y_test_i)
        for y_pred_i, y_test_i in zip(y_pred, y_test)
    ]) / len(y_test)

    print(f'Predicted labels: {y_pred}')

    print(f'Accuracy: {accuracy * 100}%')


if __name__ == '__main__':
    main()