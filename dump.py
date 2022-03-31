import numpy as np
import tifffile as tf
import imageio
import matplotlib.pyplot as plt
import time
from matplotlib.animation import ArtistAnimation, FuncAnimation
import scipy.special as special

plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\goubi\ffmpeg-5.0-essentials_build\bin\ffmpeg.exe'


def cov(X, Y):
    X = np.ravel(X)
    Y = np.ravel(Y)
    return np.mean((X - np.mean(X)) * (Y - np.mean(Y)))


if __name__ == '__main__':
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
    #W = np.broadcast_to(W, (nb, *shape)).transpose((1, 2, 0))
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
    #bessel = special.b
    plt.xlabel(r"Pupil position lag normalized to pupil diameter")
    plt.ylabel(r"Correlation coefficient $\rho(I_0, I_\tau)$")
    plt.legend()
    plt.show()
    exit()
    nb = 50
    shape = (600, 600)
    tau_s = np.linspace(0, 1.85, nb)
    tau_c = 0.37
    rho_s = np.exp(-tau_s / tau_c)
    r_s = np.sqrt(rho_s)
    x = np.linspace(0, 2 * np.pi, nb)
    r_s = ((np.sin(x) + 1) / 2)
    plt.plot(x, r_s)
    plt.show()
    M1 = np.exp(-1j * np.random.uniform(-np.pi, np.pi, shape))
    M2 = np.exp(-1j * np.random.uniform(-np.pi, np.pi, shape))
    r_s_mat = np.ones((*shape, nb))
    r_s_mat[150:451, ...] = np.multiply.outer(r_s_mat[150:451, :, 0], r_s)
    M1 = np.broadcast_to(M1, r_s_mat.transpose((2, 0, 1)).shape).transpose((1, 2, 0))
    M2 = np.broadcast_to(M2, r_s_mat.transpose((2, 0, 1)).shape).transpose((1, 2, 0))
    W = r_s_mat * M1 + np.sqrt(1 - r_s_mat ** 2) * M2
    Y, X = np.indices(shape)
    Y -= shape[0] // 2
    X -= shape[0] // 2
    mask = (X ** 2 + Y ** 2 - 25 ** 2) <= 0
    f = lambda matrix: np.abs(np.fft.ifft2(np.fft.fft2(matrix) * mask)) ** 2
    I_s = [f(W[:, :, i]).real for i in range(nb)]
    I_s = [I_s[i] / np.max(I_s[i]) for i in range(len(I_s))]
    fig, ax = plt.subplots()
    ims = [[ax.imshow(I, cmap="gray")] for I in I_s]
    ani = ArtistAnimation(fig, ims, interval=50, blit=True, repeat=False)
    ani.save("test_sine_r.mp4")
    plt.show()
    rho_s = []
    for i in range(nb):
        X = I_s[0][150:451, ...]
        Y = I_s[i][150:451, ...]
        rho = cov(X, Y) / (np.std(X) * np.std(Y))
        rho_s.append(rho)
    plt.scatter(range(nb), rho_s)
    plt.xlabel(r"Step")
    plt.ylabel(r"Correlation coefficient $\rho(I_0, I_\tau)$")
    # plt.plot(range(nb), 1 - r_s ** 0.5)
    plt.show()
    exit()
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

    exit()
    nFrames = 175
    dists = [np.random.normal(0, 2, 1000 * 1000) for _ in range(nFrames)]
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(dists[0], 256)
    global previous_max_y, previous_max_x, previous_min_x
    previous_max_y = max(n)
    previous_max_x = max(bins)
    previous_min_x = min(bins)
    ax.set_ylim(top=previous_max_y)
    ax.set_xlim(previous_min_x - 0.5 * np.abs(previous_min_x), previous_max_x + 0.5 * np.abs(previous_min_x))
    ax.set_title(f"Histogram of time step {0}")


    def update_hist(i):
        ax.clear()
        n, bins, patches = ax.hist(dists[i], 256)

        current_max_y = max(n)
        current_max_x = max(bins)
        current_min_x = min(bins)
        global previous_max_y, previous_max_x, previous_min_x
        if previous_max_y < current_max_y:
            previous_max_y = current_max_y
        if current_max_x > previous_max_x:
            previous_max_x = current_max_x
        if current_min_x < previous_min_x:
            previous_min_x = current_min_x
        ax.set_ylim(top=previous_max_y)
        ax.set_xlim(previous_min_x - 0.5 * np.abs(previous_min_x), previous_max_x + 0.5 * np.abs(previous_min_x))
        ax.set_title(f"Histogram of time step {i}")
        return patches  # .patches


    anim = FuncAnimation(fig, update_hist, frames=nFrames, blit=False, repeat=False)
    plt.show()
    anim.save("test_bar.mp4")
