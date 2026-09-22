"""Fixed-point search utilities for leaky mRNN dynamics."""

import torch
import numpy as np
import time
from copy import deepcopy

from rnntoolkit.fixed_points.fp import FixedPointCollection
from rnntoolkit.fixed_points.fp_finder import FixedPointFinderBase
from mrnntorch.mrnn.leaky_mrnn import mRNN
from mrnntorch.mrnn.elman_mrnn import ElmanmRNN
from mrnntorch.analysis.adapter import mRNNAdapter
import warnings


class mFixedPointFinder(FixedPointFinderBase[mRNN]):
    """Fixed-point finder specialized for leaky :class:`mRNN` dynamics."""

    def __init__(
        self,
        rnn: mRNN | ElmanmRNN,
        lr_init: float = 1e-4,
        tol_q: float = 1e-12,
        tol_dq: float = 1e-20,
        max_iters: int = 5000,
        do_rerun_q_outliers: bool = False,
        outlier_q_scale: float = 10.0,
        do_exclude_distance_outliers: bool = True,
        outlier_distance_scale: float = 10.0,
        tol_unique: float = 1e-3,
        max_n_unique: int | float = np.inf,
        dtype: str = "float32",
        random_seed: int = 0,
        verbose: bool = True,
        super_verbose: bool = False,
        n_iters_per_print_update: int = 100,
    ):
        """Initialize fixed-point search hyperparameters for a leaky mRNN.

        Args:
            rnn (mRNN): Network whose fixed points will be optimized.
            lr_init (float): Initial optimizer learning rate.
            tol_q (float): Absolute fixed-point objective tolerance.
            tol_dq (float): Per-step objective improvement tolerance.
            max_iters (int): Maximum optimization iterations.
            do_rerun_q_outliers (bool): Whether to rerun optimization on high-q outliers.
            outlier_q_scale (float): Multiplier used to classify q outliers.
            do_exclude_distance_outliers (bool): Whether to discard distant fixed points.
            outlier_distance_scale (float): Distance threshold scale for outlier removal.
            tol_unique (float): Tolerance used to collapse duplicate fixed points.
            max_n_unique (int): Maximum number of unique fixed points to retain.
            dtype (str): Torch dtype name used during optimization.
            random_seed (int): Random seed for reproducible sampling.
            verbose (bool): Whether to print high-level progress.
            super_verbose (bool): Whether to print per-iteration progress.
            n_iters_per_print_update (int): Iteration interval between progress prints.
        """
        super().__init__(
            rnn,
        )

        self.dtype = dtype
        self.device = next(rnn.parameters()).device
        self.torch_dtype = getattr(torch, self.dtype)
        self.adapter = mRNNAdapter(rnn)

        # Make random sequences reproducible
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

        # *********************************************************************
        # Optimization hyperparameters ****************************************
        # *********************************************************************

        self.lr_init = lr_init
        self.tol_q = tol_q
        self.tol_dq = tol_dq
        self.max_iters = max_iters
        self.do_rerun_q_outliers = do_rerun_q_outliers
        self.outlier_q_scale = outlier_q_scale
        self.do_exclude_distance_outliers = do_exclude_distance_outliers
        self.outlier_distance_scale = outlier_distance_scale
        self.tol_unique = tol_unique
        self.max_n_unique = max_n_unique
        self.verbose = verbose
        self.super_verbose = super_verbose
        self.n_iters_per_print_update = n_iters_per_print_update

    # *************************************************************************
    # Primary exposed functions ***********************************************
    # *************************************************************************

    def region_initial_states(
        self,
        states: torch.Tensor,
        n_inits: int,
        *args,
        static_state_t: int | None = None,
        noise_scale: float = 0.0,
        excluded_static_regions: list = [],
    ):

        states = self._broadcast_nxd(states, tile_n=1)

        # get included region states
        included_region_states = self.rnn.get_region_activity(states, *args)
        init_states = self.sample_states(
            included_region_states, n_inits=n_inits, noise_scale=noise_scale
        )

        # get static regions states
        non_static_regions = [*args, *excluded_static_regions]
        static_regions = self.rnn.get_excluded_hid_regions(*non_static_regions)
        if len(static_regions) != 0 and static_state_t is not None:
            static_region_states = self.rnn.get_region_activity(states, *static_regions)
            static_region_states = static_region_states[static_state_t].repeat(
                n_inits, 1
            )
        elif len(static_regions) != 0 and static_state_t is None:
            static_region_states = self.rnn.get_region_activity(states, *static_regions)
            static_region_states = self.sample_states(
                static_region_states, n_inits=n_inits, noise_scale=noise_scale
            )

        # get excluded region states (zeros)
        if len(excluded_static_regions) != 0:
            excluded_region_size = self.rnn.get_region_sizes(*excluded_static_regions)
            excluded_region_states = torch.zeros(size=(n_inits, excluded_region_size))

        # combine all states in correct order
        if len(static_regions) != 0:
            init_states = self.rnn.combine_states(
                init_states, static_region_states, list(args), static_regions
            )
        if len(excluded_static_regions) != 0:
            init_states = self.rnn.combine_states(
                init_states,
                excluded_region_states,
                [*args, *static_regions],
                excluded_static_regions,
            )

        return init_states

    def find_fixed_points(
        self,
        initial_states: torch.Tensor,
        ext_inputs: torch.Tensor,
        *args,
        optimize_h: bool = False,
        stim_inp: torch.Tensor | None = None,
        W_rec: torch.Tensor | None = None,
        n_rounds_q_opt: int = 1,
    ) -> tuple[FixedPointCollection, FixedPointCollection]:
        """Finds RNN fixed points and the Jacobians at the fixed points.

        Args:
            initial_states: Tensor specifying the initial
            states of the RNN, from which the optimization will search for
            fixed points.

            ext_inputs: external inputs to the RNN
            stim_inp: Additional stimulus input to the network
            W_rec: Fixed weight matrix to replace self.mrnn.W_rec in forward pass
            W_rec: Fixed weight matrix to replace self.mrnn.W_inp in forward pass
            n_rounds_q_opt: Number of rounds to run extra iterations on q outliers

        Returns:
            unique_fps: A FixedPoints object containing the set of unique
            fixed points after optimizing from all initial_states. Two fixed
            points are considered unique if all absolute element-wise
            differences are less than tol_unique AND the corresponding inputs
            are unique following the same criteria. See FixedPoints.py for
            additional detail.

            all_fps: A FixedPoints object containing the likely redundant set
            of fixed points (and associated metadata) resulting from ALL
            initializations in initial_states (i.e., the full set of fixed
            points before filtering out putative duplicates to yield
            unique_fps).
        """

        if optimize_h and not self.adapter.is_leaky:
            warnings.warn("optimize_h is True but mrnn is not leaky, parameter ignored")

        all_fps = self._fp_optimization(
            initial_states,
            ext_inputs,
            *args,
            optimize_h=optimize_h,
            stim_inp=stim_inp,
            W_rec=W_rec,
        )

        # Filter out duplicates after from the first optimization round
        # If optimization is performed on h, get unique using Fxstar
        # this is because Fxstar is h_next, so unique will be performed on activation
        # This is a workaround, however keeping xstar as x is good for
        # when a user might want to pass the fixed point to the mrnn again (i.e. during linearization)
        unique_fps = all_fps.get_unique(
            use_F_xstar=self.adapter.is_leaky and optimize_h
        )

        self._print_if_verbose("\tIdentified %d unique fixed points." % unique_fps.n)

        if self.do_exclude_distance_outliers:
            unique_fps = self._exclude_distance_outliers(unique_fps, initial_states)

        # Optionally run additional optimization iterations on identified
        # fixed points with q values on the large side of the q-distribution.
        if self.do_rerun_q_outliers:
            unique_fps = self._run_additional_iterations_on_outliers(
                unique_fps,
                *args,
                optimize_h=optimize_h,
                stim_inp=stim_inp,
                W_rec=W_rec,
                n_rounds=n_rounds_q_opt,
            )
            # Filter out duplicates after from the second optimization round
            unique_fps = unique_fps.get_unique(
                use_F_xstar=self.adapter.is_leaky and optimize_h
            )

        # Optionally subselect from the unique fixed points (e.g., for
        # computational savings when not all are needed.)
        if unique_fps.n > self.max_n_unique:
            self._print_if_verbose(
                "\tRandomly selecting %d unique "
                "fixed points to keep." % self.max_n_unique
            )
            max_n_unique = int(self.max_n_unique)
            idx_keep = list(self.rng.choice(unique_fps.n, max_n_unique, replace=False))
            unique_fps = unique_fps[idx_keep]

        self._print_if_verbose("\tFixed point finding complete.\n")

        return unique_fps, all_fps

    # *************************************************************************
    # Helper functions ********************************************************
    # *************************************************************************

    def _run_additional_iterations_on_outliers(
        self,
        fps: FixedPointCollection,
        *args,
        optimize_h: bool = False,
        n_rounds: int = 1,
        stim_inp: torch.Tensor | None = None,
        W_rec: torch.Tensor | None = None,
    ) -> FixedPointCollection:
        """Rerun optimization on candidate fixed points with unusually large q.

        Args:
            fps (FixedPointCollection): Candidate fixed points from a prior run.
            *args (str): Optional recurrent regions to optimize.
            n_rounds (int): Number of rerun rounds to perform.
            stim_inp (torch.Tensor | None): Optional stimulus input during reruns.
            W_rec (torch.Tensor | None): Optional recurrent weight matrix override.

        Returns:
            FixedPointCollection: Updated fixed-point collection.
        """

        """
        Known issue:
            Additional iterations do not always reduce q! This may have to do
            with learning rate schedules restarting from values that are too
            large.
        """

        assert fps.qstar is not None

        outlier_min_q = float(np.median(fps.qstar) * self.outlier_q_scale)

        def perform_outlier_optimization(
            fps: FixedPointCollection,
        ) -> FixedPointCollection:
            """Optimize only the currently identified q outliers."""
            idx_outliers = self.identify_q_outliers(fps, outlier_min_q)

            outlier_fps = fps[idx_outliers.tolist()]
            n_prev_iters = outlier_fps.n_iters
            inputs = outlier_fps.inputs
            initial_states = outlier_fps.xstar

            self._print_if_verbose(
                "\tPerforming another round of "
                "joint optimization, "
                "over outlier states only."
            )

            assert inputs is not None
            assert n_prev_iters is not None

            updated_outlier_fps = self._fp_optimization(
                initial_states,
                inputs,
                *args,
                optimize_h=optimize_h,
                stim_inp=stim_inp,
                W_rec=W_rec,
            )

            assert updated_outlier_fps.n_iters is not None

            updated_outlier_fps.n_iters += n_prev_iters
            fps[idx_outliers.tolist()] = updated_outlier_fps

            return fps

        def outlier_update(fps: FixedPointCollection) -> torch.Tensor:
            """Identify the current q outliers and report their count."""
            idx_outliers = self.identify_q_outliers(fps, outlier_min_q)
            n_outliers = len(idx_outliers)

            self._print_if_verbose(
                "\n\tDetected %d putative outliers "
                "(q>%.2e)." % (n_outliers, outlier_min_q)
            )

            return idx_outliers

        idx_outliers = outlier_update(fps)

        if len(idx_outliers) == 0:
            return fps

        for _ in range(n_rounds):
            fps = perform_outlier_optimization(fps)
            idx_outliers = outlier_update(fps)
            if len(idx_outliers) == 0:
                return fps

        return fps

    def _fp_optimization(
        self,
        initial_states: torch.Tensor,
        ext_inp: torch.Tensor,
        *args,
        optimize_h: bool = False,
        stim_inp: torch.Tensor | None = None,
        W_rec: torch.Tensor | None = None,
    ) -> FixedPointCollection:
        """Optimize a batch of candidate states toward fixed points jointly.

        Args:
            initial_states (torch.Tensor): Initial recurrent states to optimize.
            ext_inp (torch.Tensor): Constant external inputs paired with each state.
            *args (str): Optional recurrent regions to optimize.
            optimize_h (bool): whether to define the loss using h_next instead of x_next
            x_l2_scalaar (float): how to scale l2 regularization on x, only used if optimize_h
            stim_inp (torch.Tensor | None): Optional stimulus input during optimization.
            W_rec (torch.Tensor | None): Optional recurrent weight matrix override.

        Returns:
            FixedPointCollection: Optimized fixed points and optimization metadata.
        """

        # Get batch and time dims
        if self.batch_first:
            TIME_DIM = 1
        else:
            TIME_DIM = 0

        initial_states = self._broadcast_nxd(initial_states, tile_n=1)

        # Get batch size of states
        n = initial_states.shape[0]

        # Broadcast external input to [n, d]
        ext_inp = self._broadcast_nxd(ext_inp, tile_n=n)

        # Broadcast stimulus input to [n, d]
        if stim_inp is not None:
            stim_inp = self._broadcast_nxd(stim_inp, tile_n=n)
        else:
            stim_inp = torch.zeros_like(initial_states)
            stim_inp = stim_inp.to(self.torch_dtype)
            stim_inp = stim_inp.to(self.device)

        # assert the correct batch shapes
        assert ext_inp.shape[0] == initial_states.shape[0]
        assert stim_inp.shape[0] == initial_states.shape[0]

        self._print_if_verbose(
            "\nSearching for fixed points from %d initial states.\n" % n
        )

        # Ensure that fixed point optimization does not alter RNN parameters.
        print(
            "\tFreezing model parameters so model is not affected by fixed point optimization."
        )

        for p in self.rnn.parameters():
            p.requires_grad = False

        self._print_if_verbose("\tFinding fixed points via joint optimization.")
        # initialize args to include all regions if empty
        if not args:
            args = list(self.rnn.region_dict.keys())

        ext_inp.requires_grad = False

        # Gather all of the regions to concatenate during training
        # Get them region by region for proper optimization
        region_tensor_list = []
        region_to_opt_idx = []
        for i, region in enumerate(self.rnn.region_dict):
            act = self.rnn.get_region_activity(initial_states, region)
            region_tensor_list.append(act)
            if region in args:
                act.requires_grad = True
                region_to_opt_idx.append(i)

        init_lr = self.lr_init
        optimizer = torch.optim.Adam(
            [region_tensor_list[idx] for idx in region_to_opt_idx], lr=self.lr_init
        )

        iter_count = 1
        iter_learning_rate = init_lr
        t_start = time.time()
        q_prev_b = torch.full((n,), float("nan"), device=self.device)

        if W_rec is not None:
            W_rec = W_rec.detach().clone()

        # Begin optimization
        while True:
            state = torch.cat(region_tensor_list, dim=-1)
            leaky_h = (
                self.rnn.activation(state)
                if self.adapter.is_leaky and optimize_h
                else None
            )
            F_x_1xbxd = self.adapter.step(
                ext_inp,
                state,
                h=leaky_h,
                stim_input=stim_inp,
                noise=False,
                W_rec=W_rec,
            )
            state_prev = leaky_h if self.adapter.is_leaky and optimize_h else state

            state_prev_r = []
            state_next_r = []
            for region in args:
                state_prev_r.append(self.rnn.get_region_activity(state_prev, region))
                state_next_r.append(self.rnn.get_region_activity(F_x_1xbxd, region))

            state_prev_r, state_next_r = (
                torch.cat(state_prev_r, dim=-1),
                torch.cat(state_next_r, dim=-1),
            )

            dx_bxd = state_prev_r - state_next_r
            q_b = 0.5 * torch.sum(torch.square(dx_bxd), dim=-1)
            q_scalar = torch.mean(q_b)
            dq_b = torch.abs(q_b - q_prev_b)

            optimizer.zero_grad()
            q_scalar.backward()

            optimizer.step()

            ev_q_b = q_b.detach().cpu()
            ev_dq_b = dq_b.detach().cpu()

            if (
                self.super_verbose
                and np.mod(iter_count, self.n_iters_per_print_update) == 0
            ):
                self._print_iter_update(
                    iter_count, t_start, ev_q_b, ev_dq_b, iter_learning_rate
                )

            if iter_count > 1 and torch.all(
                torch.logical_or(
                    ev_dq_b < self.tol_dq * iter_learning_rate, ev_q_b < self.tol_q
                )
            ):
                """Here dq is scaled by the learning rate. Otherwise very
                small steps due to very small learning rates would spuriously
                indicate convergence. This scaling is roughly equivalent to
                measuring the gradient norm."""
                self._print_if_verbose("\tOptimization complete to desired tolerance.")
                break

            if iter_count + 1 > self.max_iters:
                self._print_if_verbose(
                    "\tMaximum iteration count reached. Terminating."
                )
                break

            q_prev_b = q_b
            iter_count += 1

        if self.verbose:
            self._print_iter_update(
                iter_count, t_start, ev_q_b, ev_dq_b, iter_learning_rate, is_final=True
            )

        # remove extra dims
        # For now make the fixed point include all regions
        full_fp = torch.cat(region_tensor_list, dim=-1)

        with torch.no_grad():
            leaky_h = (
                self.rnn.activation(full_fp)
                if self.adapter.is_leaky and optimize_h
                else None
            )
            F_x_1xbxd = self.adapter.step(
                ext_inp,
                full_fp,
                h=leaky_h,
                stim_input=stim_inp,
                noise=False,
                W_rec=W_rec,
            )

        xstar = full_fp.detach().cpu()
        F_xstar = F_x_1xbxd.detach().cpu()

        # Indicate same n_iters for each initialization (i.e., joint optimization)
        n_iters = torch.tile(torch.tensor([iter_count]), dims=(F_xstar.shape[0],))

        fps = FixedPointCollection(
            xstar=xstar,
            x_init=initial_states,
            inputs=ext_inp,
            F_xstar=F_xstar,
            qstar=ev_q_b,
            dq=ev_dq_b,
            n_iters=n_iters,
            tol_unique=self.tol_unique,
            dtype=self.torch_dtype,
        )

        return fps

    def _exclude_distance_outliers(
        self, fps: FixedPointCollection, initial_states: torch.Tensor
    ) -> FixedPointCollection:
        """Drop fixed points that are too far from the supplied initial states."""

        idx_keep = self.get_fp_non_distance_outliers(
            fps, initial_states, self.outlier_distance_scale
        )
        return fps[idx_keep.tolist()]
