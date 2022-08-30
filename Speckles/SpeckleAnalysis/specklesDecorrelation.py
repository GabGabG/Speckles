import matplotlib.pyplot as plt
import numpy as np
import os
import re
import multiprocessing as mp
import tifffile
import pandas as pd


def correlation_bunch(images):
    image_X = tifffile.imread(images[0])
    correlations = []
    for i in range(len(images)):
        image_Y = tifffile.imread(images[i])
        # print(f"Correlation {images[0]} with {images[i]}")
        correlations.append(correlation(image_X, image_Y))
    print(f"{os.getpid()} Done!")
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
        path = "/Volumes/Goliath/jroussel/Speckle/20220818-DecorrelationChickenSkin/"
        name = "20220818-DecorrelationChickenSkinPosition2Reflected"
        path += name
    else:
        path = r"C:\Users\goubi\Desktop\\"
        name = "images"
        path += name
    all_files = sortedAlphanumeric(os.listdir(path))
    all_files = [os.path.join(path, i) for i in all_files if i.endswith(".tiff")]
    b_size = 50
    splitted_files = splitContainer(all_files, b_size)
    with mp.Pool(None) as pool:
        correlations = pool.map(correlation_bunch, splitted_files)
    correlations = np.array(correlations).T
    df = pd.DataFrame(correlations)
    df.to_csv(f"{name}_batch_of_{b_size}.csv")
