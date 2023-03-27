import neurotorch as nt  # pip install git+https://github.com/NeuroTorch/NeuroTorch
from neurotorch.callbacks.base_callback import BaseCallback
import pythonbasictools as pbt  # pip install git+https://github.com/JeremieGince/PythonBasicTools
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from dynamicSpeckleSimulations import DynamicSpeckleSimulationsFromCircularSourceWithBrownianMotion
from typing import Tuple
import warnings


class GenNewDataCallback(BaseCallback):
    def on_iteration_end(self, trainer, **kwargs):
        train_dataloader = trainer.state.objects.get("train_dataloader", None)
        if train_dataloader is None:
            return

        train_dataset = train_dataloader.dataset
        train_dataset.generate_new_data()


class SpecklesTorchDataset(Dataset):
    def __init__(
            self,
            dataset_length: int = 1_000,
            n_speckles_image_per_sum: int = 4,
            t_steps: int = 50,
            nb_workers=-2,
    ):
        super().__init__()
        self.dataset_length = int(dataset_length)
        self.t_steps = t_steps
        self.n_speckles_image_per_sum = n_speckles_image_per_sum
        self.nb_workers = nb_workers
        self.data = []
        self.generate_new_data()

    def __len__(self):
        return int(self.dataset_length)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[item]

    def generate_new_data(self):
        if self.nb_workers == 0:
            self.data = [self.generate_single_speckle_seq() for _ in range(len(self))]
        else:
            self.data = pbt.multiprocessing.apply_func_multiprocess(
                func=self.generate_single_speckle_seq,
                iterable_of_args=[() for _ in range(len(self))],
                nb_workers=self.nb_workers,
                verbose=True,
            )
        return self.data

    def __distribution(self, image_flat: torch.Tensor, eigenvals: np.ndarray):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            inv_eigenvals = np.where(eigenvals >= 1e-8, 1 / eigenvals, 1e6)
        inv_eigenvals = torch.tensor(inv_eigenvals)
        diag0 = torch.diag_embed(-inv_eigenvals, offset=0)
        diag1 = torch.diag_embed(inv_eigenvals[:-1], offset=1)
        theta = diag0 + diag1
        alpha = torch.zeros_like(inv_eigenvals)
        alpha[0] = 1
        ones = torch.ones_like(alpha)
        dist = -alpha @ torch.matrix_exp(torch.einsum('p,qr->pqr', image_flat, theta)) @ theta @ ones
        return dist.type(torch.float32)

    def generate_single_speckle_seq(self):
        specks_obj = DynamicSpeckleSimulationsFromCircularSourceWithBrownianMotion(500, self.t_steps, 150, 0, 1, 0.3)

        W = specks_obj.simulate()
        speckles = specks_obj.previous_simulations
        indices = np.arange(self.t_steps)
        # print(indices)
        # print(self.n_speckles_image_per_sum)
        choice_indices = np.random.choice(indices, self.n_speckles_image_per_sum, False)
        W = W[..., choice_indices]
        speckles = speckles[choice_indices]
        covariance = np.cov(W.reshape(-1, W.shape[-1]), rowvar=False)
        eigenvals = np.linalg.eigvalsh(covariance)
        eigenvals[eigenvals < 1e-15] = 0
        image_sum = np.sum(speckles, 0)
        image_sum_flat = torch.tensor(image_sum.flatten(), dtype=torch.float32)
        distribution = self.__distribution(image_sum_flat, eigenvals)
        return image_sum_flat, distribution


class SpeckleDistribution(torch.nn.Module):
    def __init__(self, theta_size: int):
        super(SpeckleDistribution, self).__init__()
        self.theta_size = theta_size
        self.alpha = torch.zeros(theta_size)
        self.alpha[0] = 1
        self.lmbda_sqrt = torch.nn.Parameter(torch.randn(self.theta_size), requires_grad=True)
        self.theta_placeholder = torch.zeros((self.theta_size, self.theta_size))

    def get_lmbda(self):
        return torch.pow(self.lmbda_sqrt, 2)

    def get_theta(self):
        lmbda = self.get_lmbda()
        diag0 = torch.diag_embed(-lmbda, offset=0)
        diag1 = torch.diag_embed(lmbda[:-1], offset=1)
        theta = diag0 + diag1
        return theta

    def forward(self, x, **kwargs):
        theta = self.get_theta()
        ones = torch.ones(self.theta_size).to(theta.device)
        dist = -self.alpha.to(theta.device) @ torch.matrix_exp(torch.einsum('bp,qr->bpqr', x, theta)) @ theta @ ones
        if kwargs.get("log", False):
            return torch.log(dist)
        return dist

    def extra_repr(self) -> str:
        with torch.no_grad():
            lmbda = self.get_lmbda()
        return f"Lambda: {lmbda}"


class MySequential(nt.Sequential):
    def get_dist(self, *args, **kwargs):
        kwargs["log"] = False
        out = self(*args, **kwargs)
        out, _ = nt.utils.unpack_out_hh(out)
        if isinstance(out, dict):
            out = out[list(out.keys())[0]]
        return out

    def get_log_dist(self, *args, **kwargs):
        kwargs["log"] = True
        out = self(*args, **kwargs)
        out, _ = nt.utils.unpack_out_hh(out)
        if isinstance(out, dict):
            out = out[list(out.keys())[0]]
        return out









class EndTraining:

    def __init__(self, model:MySequential, test_dataset: SpecklesTorchDataset):
        self.model = model
        self.test_dataset = test_dataset

    def visualize(self):
        data = self.test_dataset.data
        x, y = data[0]
        pred = model.get_dist(torch.unsqueeze(x))


if __name__ == '__main__':
    batch_size = 1
    dataloaders = dict(
        train=DataLoader(
            SpecklesTorchDataset(dataset_length=batch_size * 1, nb_workers=4),
            batch_size=batch_size, shuffle=True, num_workers=0
        ),
        val=DataLoader(
            SpecklesTorchDataset(dataset_length=batch_size, nb_workers=4),
            batch_size=batch_size, shuffle=False, num_workers=0
        ),
    )
    # TODO: Faire un github issue de l'erreur lorsque nt.Sequential est utilisé à la place de MySequential.
    model = MySequential(
        layers=[SpeckleDistribution(theta_size=4)],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        checkpoint_folder="./checkpoints/checkpoint_000",
    )
    checkpoint_manager = nt.CheckpointManager(model.checkpoint_folder)
    learning_algorithm = nt.BPTT(
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-2),
        criterion=torch.nn.KLDivLoss(),
    )
    trainer = nt.Trainer(
        model=model,
        predict_method="get_log_dist",
        callbacks=[checkpoint_manager, learning_algorithm],
        verbose=True
    )
    trainer_str = str(trainer).replace('\n', '\n\t')
    print(f"Trainer:\n\t{trainer_str}")
    training_history = trainer.train(
        dataloaders["train"],
        dataloaders["val"],
        n_iterations=100,
        load_checkpoint_mode=nt.LoadCheckpointMode.LAST_ITR,
        force_overwrite=True,
    )
    training_history.plot(show=True)
    model.load_checkpoint(checkpoint_manager.checkpoints_meta_path, nt.LoadCheckpointMode.BEST_ITR, verbose=True)
