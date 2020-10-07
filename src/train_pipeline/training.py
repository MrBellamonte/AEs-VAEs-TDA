"""Training classes."""
import time

import torch
from torch.utils.data import DataLoader

from src.models.WitnessComplexAE.train_util import compute_wc_offline
from src.topology.witness_complex import WitnessComplex


class TrainingLoop():
    """Training a model using a dataset."""

    def __init__(self, model, dataset, n_epochs, batch_size, learning_rate, method_args = None,
                 weight_decay=1e-5, device='cuda', callbacks=None, verbose = False):
        """Training of a model using a dataset and the defined callbacks.

        Args:
            model: AutoencoderModel
            dataset: Dataset
            n_epochs: Number of epochs to train
            batch_size: Batch size
            learning_rate: Learning rate
            callbacks: List of callbacks
        """
        self.model = model
        self.dataset = dataset
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.callbacks = callbacks if callbacks else []
        self.verbose = verbose

        if method_args == None:
            self.method_args = dict(name = None)
        else:
            self.method_args = method_args

    def _execute_callbacks(self, hook, local_variables):
        stop = False
        for callback in self.callbacks:
            # Convert return value to bool --> if callback doesn't return
            # anything we interpret it as False
            stop |= bool(getattr(callback, hook)(**local_variables))
        return stop

    def on_epoch_begin(self, local_variables):
        """Call callbacks before an epoch begins."""
        return self._execute_callbacks('on_epoch_begin', local_variables)

    def on_epoch_end(self, local_variables):
        """Call callbacks after an epoch is finished."""
        return self._execute_callbacks('on_epoch_end', local_variables)

    def on_batch_begin(self, local_variables):
        """Call callbacks before a batch is being processed."""
        self._execute_callbacks('on_batch_begin', local_variables)

    def on_batch_end(self, local_variables):
        """Call callbacks after a batch has be processed."""
        self._execute_callbacks('on_batch_end', local_variables)

    # pylint: disable=W0641
    def __call__(self):
        """Execute the training loop."""
        model = self.model
        dataset = self.dataset
        n_epochs = self.n_epochs
        batch_size = self.batch_size
        learning_rate = self.learning_rate

        n_instances = len(dataset)

        # the dataset. This is necassary because the surrogate approach does
        # not yet support changes in the batch dimension.
        if self.method_args['name'] == 'topoae_wc':
            print('WARNING: drop last batch if not complete!')
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                      pin_memory=True, drop_last=True)
        else:
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                      pin_memory=True, drop_last=False)
        n_batches = len(train_loader)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate,
            weight_decay=self.weight_decay)





        if self.method_args['name'] == 'topoae_wc':
            dist_X_all, pair_mask_X_all = compute_wc_offline(dataset, train_loader, batch_size, self.method_args, name='Training Dataset', verfication = True)


        #mu = 0.5
        run_times_epoch = []
        for epoch in range(1, n_epochs+1):
            #mu = 0.1*max(0,(int(n_epochs/2)-epoch)/int(n_epochs/2))
            if self.on_epoch_begin(remove_self(locals())):
                break
            t_start = time.time()
            for batch, (img, label) in enumerate(train_loader):


                self.on_batch_begin(remove_self(locals()))

                # Set model into training mode and compute loss
                model.train()

                if self.method_args['name'] == 'topoae_wc':
                    x = img.to(self.device)
                    dist_X = dist_X_all[batch, :, :].to(self.device)
                    pair_mask_X = pair_mask_X_all[batch, :, :].to(self.device)
                    if (batch == 1) and (epoch in [5,10,15,20,25,30,35,40]):
                        l = label
                    else:
                        l = None
                    loss, loss_components = self.model(x, dist_X, pair_mask_X, labels=l)
                else:
                    x = img.to(self.device)
                    loss, loss_components = self.model(x.float())
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Call callbacks
                self.on_batch_end(remove_self(locals()))
            t_end = time.time()
            run_times_epoch.append((t_end-t_start))
            if self.verbose:
                print('TIME: {}'.format(t_end-t_start))
            if self.on_epoch_end(remove_self(locals())):
                break
        return epoch, run_times_epoch


def remove_self(dictionary):
    """Remove entry with name 'self' from dictionary.

    This is useful when passing a dictionary created with locals() as kwargs.

    Args:
        dictionary: Dictionary containing 'self' key

    Returns:
        dictionary without 'self' key

    """
    del dictionary['self']
    return dictionary

