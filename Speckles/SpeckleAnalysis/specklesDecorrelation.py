import matplotlib.pyplot as plt
import numpy as np
import os
import re
import multiprocessing as mp
import tifffile


def correlation_bunch(images):
    image_X = tifffile.imread(images[0])
    correlations = []
    for i in range(len(images)):
        image_Y = tifffile.imread(images[i])
        # print(f"Correlation {images[0]} with {images[i]}")
        correlations.append(correlation(image_X, image_Y))
    return correlations


def correlation(X, Y):
    X = np.ravel(X)
    Y = np.ravel(Y)
    mu_X = X.mean()
    mu_Y = Y.mean()
    std_x = X.std()
    std_y = Y.std()
    return np.mean((X - mu_X) * (Y - mu_Y)) / (std_x * std_y)


def sortedAlphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanumKey = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanumKey)


def splitContainer(container, n_elements):
    return [container[i:i + n_elements] for i in range(0, len(container), n_elements)]


def squared(numbers):
    ret = []
    for i in numbers:
        ret.append(i ** 2)
    return ret


if __name__ == '__main__':
    machine = input("What machine? (local or imaris)")
    if machine == "imaris":
        path = "/Volumes/Goliath/jroussel/Speckle/20220818-DecorrelationChickenSkin/" \
               "20220818-DecorrelationChickenSkinPosition2Reflected"
        print(os.listdir(path)[:10])
    exit()
    # n = range(1000)
    # with mp.Pool(10) as pool:
    #     output = pool.map(squared, n)
    # print(output)
    # exit()
    path = r"C:\Users\goubi\Desktop\images"
    all_files = sortedAlphanumeric(os.listdir(path))
    all_files = [os.path.join(path, i) for i in all_files]
    splitted_files = splitContainer(all_files, 30)
    with mp.Pool(10) as pool:
        correlations = pool.map(correlation_bunch, splitted_files)
    for corr in correlations:
        plt.plot(corr)
    plt.show()
    # print(correlations)
