import numpy as np
import tifffile as tf
import imageio
import matplotlib.pyplot as plt
import time
from matplotlib.animation import ArtistAnimation, FuncAnimation
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.signal import convolve2d, convolve
from scipy.ndimage import gaussian_filter

plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\goubi\ffmpeg-5.0-essentials_build\bin\ffmpeg.exe'


def cov(X, Y):
    X = np.ravel(X)
    Y = np.ravel(Y)
    return np.mean((X - np.mean(X)) * (Y - np.mean(Y)))


def moving_pupil():
    shape = (1000, 1000)
    radius = 50
    Y, X = np.indices(shape)
    Y -= shape[0] // 2
    X -= shape[0] // 2
    mask = lambda center_x, center_y: ((X - center_x) ** 2 + (Y - center_y) ** 2 - radius ** 2) <= 0
    nb = 50
    pos_x = np.linspace(0, radius / 0.5, nb)
    masks = [mask(pos_x[i], 0) for i in range(nb)]
    fig, ax = plt.subplots()
    ims = [[ax.imshow(mask)] for mask in masks]
    ani = ArtistAnimation(fig, ims, interval=50, blit=True, repeat=False)
    ani.save("moving_pupil.mp4")
    plt.show()
    W = np.exp(-1j * np.random.uniform(-np.pi, np.pi, shape))
    # W = np.broadcast_to(W, (nb, *shape)).transpose((1, 2, 0))
    f = lambda matrix, mask: np.abs(np.fft.ifft2(np.fft.fft2(matrix) * mask)) ** 2
    I_s = [f(W, masks[i]).real for i in range(nb)]
    I_s = [I_s[i] / np.max(I_s[i]) for i in range(len(I_s))]
    fig, ax = plt.subplots()
    ims = [[ax.imshow(I, cmap="gray")] for I in I_s]
    ani = ArtistAnimation(fig, ims, interval=50, blit=True, repeat=False)
    ani.save("test_moving_mask.mp4")
    plt.show()
    rho_s = []
    rho_masks = []
    for i in range(nb):
        X = I_s[0]
        Y = I_s[i]
        rho = cov(X, Y) / (np.std(X) * np.std(Y))
        rho_s.append(rho)
        X = masks[0]
        Y = masks[i]
        rho = cov(X, Y) / (np.std(X) * np.std(Y))
        rho_masks.append(rho)
    plt.scatter(pos_x / (radius * 2), rho_s, label="Coefficients de corrélation des speckles")
    msg = "Coefficients de corrélation (au carré) du mouvement de la pupille"
    plt.plot(pos_x / (radius * 2), np.power(rho_masks, 2), color="red", label=msg)
    plt.xlabel(r"Pupil position lag normalized to pupil diameter")
    plt.ylabel(r"Correlation coefficient $\rho(I_0, I_\tau)$")
    plt.legend()
    plt.show()


def decorrelation_following_function():
    nb = 50
    shape = (600, 600)
    x = np.linspace(0, 2 * np.pi, nb)
    r_s = np.linspace(0, 1, 50)[::-1]
    plt.plot(x, r_s)
    plt.show()
    M1 = np.exp(-1j * np.random.uniform(-np.pi, np.pi, shape))
    M2 = np.exp(-1j * np.random.uniform(-np.pi, np.pi, shape))
    r_s_mat = np.ones(shape)
    r_s_mat = np.multiply.outer(r_s_mat, r_s)
    M1 = np.broadcast_to(M1, r_s_mat.transpose((2, 0, 1)).shape).transpose((1, 2, 0))
    M2 = np.broadcast_to(M2, r_s_mat.transpose((2, 0, 1)).shape).transpose((1, 2, 0))
    W = r_s_mat * M1 + np.sqrt(1 - r_s_mat ** 2) * M2
    Y, X = np.indices(shape)
    Y -= shape[0] // 2
    X -= shape[0] // 2
    mask = (X ** 2 + Y ** 2 - 25 ** 2) <= 0
    f = lambda matrix: np.abs(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(matrix)) * mask))) ** 2
    I_s = [f(W[:, :, i]).real for i in range(nb)]
    I_s = [I_s[i] / np.max(I_s[i]) for i in range(len(I_s))]
    fig, ax = plt.subplots()
    ims = [[ax.imshow(I, cmap="gray")] for I in I_s]
    ani = ArtistAnimation(fig, ims, interval=50, blit=True, repeat=False)
    ani.save("test_sine_r.mp4")
    plt.show()
    rho_s = []
    for i in range(nb):
        X = I_s[0]
        Y = I_s[i]
        rho = cov(X, Y) / (np.std(X) * np.std(Y))
        rho_s.append(rho)
    rho_s = np.array(rho_s)
    plt.scatter(r_s, rho_s, color="red", marker="x")
    plt.xlabel(r"Correlation coefficient between initial phase matrices $r$")
    plt.ylabel(r"Correlation coefficient $\rho(I_0, I_\tau)$")
    plt.plot(r_s, r_s ** 2)
    plt.show()


def brownian():
    tau_s = np.arange(0, 3, 0.037)
    tau_c = 0.8
    rho_s_th = np.exp(-tau_s / tau_c)
    r_s = np.sqrt(rho_s_th)
    n = len(r_s)
    shape = 600
    M1 = np.exp(-1j * np.random.uniform(-np.pi, np.pi, (shape, shape)))
    M2 = np.exp(-1j * np.random.uniform(-np.pi, np.pi, (shape, shape)))
    W = np.multiply.outer(M1, r_s) + np.multiply.outer(M2, np.sqrt(1 - r_s ** 2))
    Y, X = np.indices((shape, shape))
    Y -= shape // 2
    X -= shape // 2
    mask = (X ** 2 + Y ** 2 - 200 ** 2) <= 0
    masks = np.dstack([mask for _ in range(n)])
    f = lambda matrix: np.abs(np.fft.ifft2(np.fft.fft2(matrix) * mask)) ** 2
    fft_mask = np.fft.fft2(mask)
    f_2 = lambda matrix: np.abs(np.fft.fftshift(np.fft.fft2(matrix))) ** 2
    I_s = [f(W[:, :, i]).real for i in range(n)]
    I_s_2 = [f_2(W[:, :, i] * mask).real for i in range(n)]
    I_s = [I_s[i] / np.max(I_s[i]) for i in range(len(I_s))]
    I_s_2 = [I_s_2[i] / np.max(I_s_2[i]) for i in range(len(I_s_2))]
    fig, ax = plt.subplots()
    ims = [[ax.imshow(I, cmap="gray")] for I in I_s]
    ani = ArtistAnimation(fig, ims, interval=50, blit=True, repeat=False)
    ani.save("test_brownian.mp4")
    plt.show()
    fig, ax = plt.subplots()
    ims = [[ax.imshow(I, cmap="gray")] for I in I_s_2]
    ani = ArtistAnimation(fig, ims, interval=50, blit=True, repeat=False)
    ani.save("test_brownian_2.mp4")
    plt.show()
    # plt.plot(r_s, r_s ** 2)
    rho_s = []
    rho_s_2 = []
    for i in range(0, n):
        X = I_s[0]
        Y = I_s[i]
        X_2 = I_s_2[0]
        Y_2 = I_s_2[i]
        rho = cov(X, Y) / (np.std(X) * np.std(Y))
        rho_2 = cov(X_2, Y_2) / (np.std(X_2) * np.std(Y_2))
        rho_s.append(rho)
        rho_s_2.append(rho_2)
    plt.scatter(tau_s, rho_s)
    plt.scatter(tau_s, rho_s_2)
    plt.plot(tau_s, rho_s_th)
    plt.xlabel(r"Time step $\tau$")
    plt.ylabel(r"Correlation coefficient $\rho(I_0, I_\tau)$")
    plt.show()


def divide_chunk(array, n):
    return [np.mean(array[i:i + n, :, :], axis=0) for i in range(0, array.shape[0], n)]


def speckling(mask: np.ndarray):
    phases = np.random.uniform(-np.pi, np.pi, mask.shape)
    phasors = np.exp(1j * phases)
    speckles = (np.abs(np.fft.fftshift(np.fft.fft2(phasors * mask))) ** 2).real
    speckles /= np.max(speckles)
    return speckles


def gamma(x, k, scale):
    return stats.gamma.pdf(x, k, scale=scale)


def exponential_fit(data):
    return stats.expon.fit(data, floc=0)


def gamma_fit(data):
    return stats.gamma.fit(data, floc=0)


def gamma_cdf(x, n, loc, theta):
    return stats.gamma.cdf(x, n, scale=theta, loc=loc)


def exponential_cdf(x, loc, scale):
    return stats.expon.cdf(x, scale=scale, loc=loc)


def gamma_pdf(x, n, loc, theta):
    return stats.gamma.pdf(x, n, scale=theta, loc=loc)


def exponential_pdf(x, loc, scale):
    return stats.expon.pdf(x, loc=loc, scale=scale)


def speckling_speckles(mask, n_speckles):
    speckles = np.full((*mask.shape, n_speckles), np.nan)
    means = []
    for i in range(n_speckles):
        current_speckles = speckling(mask)
        speckles[:, :, i] = current_speckles
        means.append(np.mean(current_speckles))
    s_speckles = np.sum(speckles, axis=-1)
    return s_speckles, means


if __name__ == '__main__':
    n = 1
    shape = (1000, 1000)
    Y, X = np.indices(shape)
    Y -= shape[0] // 2
    X -= shape[1] // 2
    mask = (X ** 2 + Y ** 2 - 100 ** 2) <= 0
    s_speckles, means = speckling_speckles(mask, n)
    data = s_speckles.ravel()
    args_gamma = gamma_fit(data)
    args_expon = exponential_fit(data)
    n_data, bins, _ = plt.hist(data, 256, None, True)
    x = (bins[:-1] + bins[1:]) / 2
    plt.plot(x, gamma_pdf(x, *args_gamma), color="green", linestyle="--", label="Gamma fit")
    plt.plot(x, exponential_pdf(x, *args_expon), color="red", linestyle=":", label="Exponential fit")
    plt.legend()
    print(args_gamma)
    res_gamma = stats.ks_1samp(data, gamma_cdf, args_gamma)
    print(f"K-S Test avec gamma : {res_gamma}")
    print(args_expon)
    res_expon = stats.ks_1samp(data, exponential_cdf, args_expon)
    print(f"K-S Test avec exponentielle : {res_expon}")
    plt.show()
    exit()
    n = 10
    shape = (1000, 1000)
    Y, X = np.indices(shape)
    Y -= shape[0] // 2
    X -= shape[1] // 2
    mask = (X ** 2 + Y ** 2 - 100 ** 2) <= 0
    speckles = np.full((*mask.shape, n), np.nan)
    means = []
    for i in range(n):
        current_speckles = speckling(mask)
        speckles[:, :, i] = current_speckles
        means.append(np.mean(current_speckles))
    print(means)
    scale = np.mean(means)
    s_speckles = np.mean(speckles, axis=-1)
    plt.imshow(s_speckles, cmap="gray")
    plt.show()
    n_, bins, _ = plt.hist(s_speckles.ravel(), bins=256, density=True)
    x = (bins[:-1] + bins[1:]) / 2
    plt.plot(x, gamma(x, n, 1 / n * scale), color="red", linestyle="--")
    plt.show()
