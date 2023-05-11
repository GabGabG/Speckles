import neurotorch as nt  # pip install git+https://github.com/NeuroTorch/NeuroTorch
from neurotorch.callbacks.base_callback import BaseCallback
import pythonbasictools as pbt  # pip install git+https://github.com/JeremieGince/PythonBasicTools
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from dynamicSpeckleSimulations import DynamicSpeckleSimulationsFromCircularSourceWithBrownianMotion
from typing import Tuple, Dict, Union, Any
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
            input_precision: int = 10,
            t_steps: int = 50,
            re_moments: bool = False,
            k_moments: int = 2,
            nb_workers=-2,
    ):
        super().__init__()
        self.dataset_length = int(dataset_length)
        self.input_precision = input_precision
        self.t_steps = t_steps
        self.re_moments = re_moments
        self.k_moments = k_moments
        self.n_speckles_image_per_sum = n_speckles_image_per_sum
        self.nb_workers = nb_workers
        self.data = []
        self.alpha = torch.zeros(self.t_steps)
        self.alpha[0] = 1
        self.generate_new_data()

    def __len__(self):
        return int(self.dataset_length)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data[item]
        if self.re_moments:
            item = item[0], item[1]['moments']
        else:
            item = item[0], item[1]['dist']
        return item

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
        # if self.re_moments:
        #     self.data = [
        #         (img, dict(**d, moments=self.compute_moments(d, self.k_moments)))
        #         for img, d in self.data
        #     ]
        return self.data

    @staticmethod
    def compute_moments(theta: torch.Tensor, k: int):
        return torch.concat([SpecklesTorchDataset.compute_moment(theta, m).unsqueeze(dim=0) for m in range(1, k + 1)])

    @staticmethod
    def compute_moment(theta: torch.Tensor, m: int):
        alpha = torch.zeros(theta.shape[0], dtype=theta.dtype)
        alpha[0] = ((-1) ** m) * np.math.factorial(m)
        theta_inv = torch.inverse(theta)
        theta_inv_m = torch.matrix_power(theta_inv, m)
        return alpha @ theta_inv_m @ torch.ones_like(alpha)

    def __distribution(self, intensity: torch.Tensor, eigenvals: np.ndarray) -> Dict[str, torch.Tensor]:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            inv_eigenvals = 1 / np.abs(eigenvals)
        inv_eigenvals = torch.tensor(inv_eigenvals)
        diag0 = torch.diag_embed(-inv_eigenvals, offset=0)
        diag1 = torch.diag_embed(inv_eigenvals[:-1], offset=1)
        theta = diag0 + diag1
        alpha = torch.zeros_like(inv_eigenvals)
        alpha[0] = 1
        ones = torch.ones_like(alpha)
        dist = -alpha @ torch.matrix_exp(torch.einsum('p,qr->pqr', intensity, theta)) @ theta @ ones
        moments = self.compute_moments(theta, self.k_moments)
        return dict(dist=dist.type(torch.float32), moments=moments)

    def generate_single_speckle_seq(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        specks_obj = DynamicSpeckleSimulationsFromCircularSourceWithBrownianMotion(500, self.t_steps, 150, 0, 1, 0.3)

        specks_obj.simulate()
        speckles = specks_obj.previous_simulations
        indices = np.arange(self.t_steps)
        # print(indices)
        # print(self.n_speckles_image_per_sum)
        choice_indices = np.random.choice(indices, self.n_speckles_image_per_sum, False)
        speckles = speckles[choice_indices]
        speckles_t = speckles.transpose(1, 2, 0)
        speckles_flat = speckles_t.reshape(-1, speckles_t.shape[-1])
        corr = np.corrcoef(speckles_flat, rowvar=False)
        means = np.mean(speckles_flat, 0)
        means = np.outer(means, means)
        eigenvals = np.linalg.eigvals(np.sqrt(means * corr))
        image_sum = np.sum(speckles, 0)
        image_sum_flat = torch.tensor(image_sum.flatten(), dtype=torch.float32)
        x = torch.linspace(0, torch.max(image_sum_flat), self.input_precision)
        distribution = self.__distribution(x, eigenvals)
        return x, distribution


class SpeckleDistribution(torch.nn.Module):
    def __init__(self, theta_size: int, **kwargs):
        super(SpeckleDistribution, self).__init__()
        self.theta_size = theta_size
        self.alpha = torch.zeros(theta_size)
        self.alpha[0] = 1
        self.lmbda_sqrt = torch.nn.Parameter(torch.randn(self.theta_size), requires_grad=True)
        self.theta_placeholder = torch.zeros((self.theta_size, self.theta_size))
        self.k_moments = kwargs.get("k_moments", 2)

    def get_lmbda(self):
        return torch.pow(self.lmbda_sqrt, 2)

    def get_theta(self):
        lmbda = self.get_lmbda()
        diag0 = torch.diag_embed(-lmbda, offset=0)
        diag1 = torch.diag_embed(lmbda[:-1], offset=1)
        theta = diag0 + diag1
        return theta

    def forward(self, x, **kwargs):
        if kwargs.get("re_moments", False):
            return self.get_theta_moments()
        theta = self.get_theta()
        ones = torch.ones(self.theta_size).to(theta.device)
        dist = -self.alpha.to(theta.device) @ torch.matrix_exp(torch.einsum('bp,qr->bpqr', x, theta)) @ theta @ ones
        if kwargs.get("log", False):
            return torch.log(dist)
        return dist

    def get_theta_moments(self, k: int = None):
        if k is None:
            k = self.k_moments
        theta = self.get_theta()
        return SpecklesTorchDataset.compute_moments(theta, k)

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

    def get_theta_moments(self, *args, **kwargs):
        kwargs["re_moments"] = True
        out = self(*args, **kwargs)
        out, _ = nt.utils.unpack_out_hh(out)
        if isinstance(out, dict):
            out = out[list(out.keys())[0]]
        return out

    def forward(  # TODO: make a PR to neurotorch to update this fonction. **kwargs is now passed to layers' call.
            self,
            inputs: Union[Dict[str, Any], torch.Tensor],
            **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        :param inputs: The inputs to the model where the dimensions are {input_name: (batch_size, input_size)}.
        :type inputs: Union[Dict[str, Any], torch.Tensor]
        :param kwargs: Additional arguments for the forward pass.

        :return: A dictionary of outputs where the values are the names of the layers and the values are the outputs
                of the layers.
        :rtype: Dict[str, torch.Tensor]
        """
        inputs = self._inputs_to_dict(inputs)
        inputs = self.apply_input_transform(inputs)
        inputs = self._format_inputs(inputs)
        outputs: Dict[str, torch.Tensor] = {}

        features_list = []
        for layer_name, layer in self.input_layers.items():
            features = layer(inputs[layer_name], **kwargs)
            features_list.append(features)
        if features_list:
            forward_tensor = torch.concat(features_list, dim=-1)
        else:
            forward_tensor = torch.concat([inputs[in_name] for in_name in inputs], dim=-1)

        for layer_idx, layer in enumerate(self.hidden_layers):
            forward_tensor = layer(forward_tensor, **kwargs)

        for layer_name, layer in self.output_layers.items():
            out = layer(forward_tensor, **kwargs)
            outputs[layer_name] = out

        outputs_tensor = self.apply_output_transform(outputs)
        return outputs_tensor


class EndTraining:

    def __init__(self, model: MySequential, test_dataset: SpecklesTorchDataset):
        self.model = model
        self.test_dataset = test_dataset
        self.dist_obj: SpeckleDistribution = self.model.get_layer()
        self.pred_theta = self.dist_obj.get_theta()

    def visualize(self):
        data = self.test_dataset.data
        x, y = data[0]
        self.model.get_dist(x)



if __name__ == '__main__':
    batch_size = 1
    _re_moments, _k_moments = False, 4
    dataloaders = dict(
        train=DataLoader(
            SpecklesTorchDataset(
                dataset_length=batch_size * 1, nb_workers=0, re_moments=_re_moments, k_moments=_k_moments
            ),
            batch_size=batch_size, shuffle=True, num_workers=0
        ),
        val=DataLoader(
            SpecklesTorchDataset(dataset_length=batch_size, nb_workers=0, re_moments=_re_moments, k_moments=_k_moments),
            batch_size=batch_size, shuffle=False, num_workers=0
        ),
    )
    # TODO: Faire un github issue de l'erreur lorsque nt.Sequential est utilisé à la place de MySequential.
    model = MySequential(
        layers=[SpeckleDistribution(theta_size=4, k_moments=_k_moments)],
        # device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        device=torch.device("cpu"),
        checkpoint_folder="./checkpoints/checkpoint_000",
    )
    checkpoint_manager = nt.CheckpointManager(model.checkpoint_folder)
    learning_algorithm = nt.BPTT(
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-2),
        # criterion=(torch.nn.MSELoss() if _re_moments else torch.nn.KLDivLoss()),
        criterion=torch.nn.MSELoss(),
    )
    trainer = nt.Trainer(
        model=model,
        predict_method=("get_theta_moments" if _re_moments else "get_dist"),
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
