# Code from normflows package (https://github.com/VincentStimper/normalizing-flows),
# adapted for conditional normalizing flows

from normflows.flows import Flow
from normflows.utils.masks import create_alternating_binary_mask
from normflows.utils.nn import PeriodicFeaturesElementwise
from normflows.nets.resnet import ResidualNet
from normflows.utils.splines import DEFAULT_MIN_DERIVATIVE
from normflows.flows.neural_spline.coupling import PiecewiseRationalQuadraticCDF
import numpy as np
import torch
import torch.nn as nn
from normflows import utils
from normflows.utils import splines
import warnings
from main.models.internal_coordinates_flows.periodic_features_cos_sin import (
    PeriodicFeaturesCosSin,
)


class Coupling(Flow):
    """A base class for coupling layers. Supports 2D inputs (NxD), as well as 4D inputs for
    images (NxCxHxW). For images the splitting is done on the channel dimension, using the
    provided 1D mask."""

    def __init__(
        self,
        mask,
        transform_net_create_fn,
        unconditional_transform=None,
        num_cond_channels=0,
    ):
        """Constructor.

        mask: a 1-dim tensor, tuple or list. It indexes inputs as follows:

        - if `mask[i] > 0`, `input[i]` will be transformed.
        - if `mask[i] <= 0`, `input[i]` will be passed unchanged.

        Args:
          mask
        """
        mask = torch.as_tensor(mask)
        if mask.dim() != 1:
            raise ValueError("Mask must be a 1-dim tensor.")
        if mask.numel() <= 0:
            raise ValueError("Mask can't be empty.")

        super().__init__()
        self.features = len(mask)
        features_vector = torch.arange(self.features)

        self.register_buffer(
            "identity_features", features_vector.masked_select(mask <= 0)
        )
        self.register_buffer(
            "transform_features", features_vector.masked_select(mask > 0)
        )

        assert self.num_identity_features + self.num_transform_features == self.features

        self.transform_net = transform_net_create_fn(
            self.num_identity_features,
            self.num_transform_features * self._transform_dim_multiplier(),
            context_features=num_cond_channels,
        )

        if unconditional_transform is None:
            self.unconditional_transform = None
        else:
            self.unconditional_transform = unconditional_transform(
                features=self.num_identity_features
            )

    @property
    def num_identity_features(self):
        return len(self.identity_features)

    @property
    def num_transform_features(self):
        return len(self.transform_features)

    def forward(self, inputs, context=None):
        if inputs.dim() not in [2, 4]:
            raise ValueError("Inputs must be a 2D or a 4D tensor.")

        if inputs.shape[1] != self.features:
            raise ValueError(
                "Expected features = {}, got {}.".format(self.features, inputs.shape[1])
            )

        identity_split = inputs[:, self.identity_features, ...]
        transform_split = inputs[:, self.transform_features, ...]

        transform_params = self.transform_net(
            identity_split, context
        )  # For now, only use the context for the conditional transform
        transform_split, logabsdet = self._coupling_transform_forward(
            inputs=transform_split, transform_params=transform_params
        )

        if self.unconditional_transform is not None:
            identity_split, logabsdet_identity = self.unconditional_transform(
                identity_split,
                None,  # TODO: Maybe at some point implement a unconditional transform that depends on the context
            )
            logabsdet += logabsdet_identity

        outputs = torch.empty_like(inputs)
        outputs[:, self.identity_features, ...] = identity_split
        outputs[:, self.transform_features, ...] = transform_split

        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if inputs.dim() not in [2, 4]:
            raise ValueError("Inputs must be a 2D or a 4D tensor.")

        if inputs.shape[1] != self.features:
            raise ValueError(
                "Expected features = {}, got {}.".format(self.features, inputs.shape[1])
            )

        identity_split = inputs[:, self.identity_features, ...]
        transform_split = inputs[:, self.transform_features, ...]

        logabsdet = 0.0
        if self.unconditional_transform is not None:
            identity_split, logabsdet = self.unconditional_transform.inverse(
                identity_split,
                None,  # TODO: Maybe at some point implement a unconditional transform that depends on the context
            )

        transform_params = self.transform_net(identity_split, context)
        transform_split, logabsdet_split = self._coupling_transform_inverse(
            inputs=transform_split, transform_params=transform_params
        )
        logabsdet += logabsdet_split

        outputs = torch.empty_like(inputs)
        outputs[:, self.identity_features] = identity_split
        outputs[:, self.transform_features] = transform_split

        return outputs, logabsdet

    def _transform_dim_multiplier(self):
        """Number of features to output for each transform dimension."""
        raise NotImplementedError()

    def _coupling_transform_forward(self, inputs, transform_params):
        """Forward pass of the coupling transform."""
        raise NotImplementedError()

    def _coupling_transform_inverse(self, inputs, transform_params):
        """Inverse of the coupling transform."""
        raise NotImplementedError()


class PiecewiseCoupling(Coupling):
    def _coupling_transform_forward(self, inputs, transform_params):
        return self._coupling_transform(inputs, transform_params, inverse=False)

    def _coupling_transform_inverse(self, inputs, transform_params):
        return self._coupling_transform(inputs, transform_params, inverse=True)

    def _coupling_transform(self, inputs, transform_params, inverse=False):
        if inputs.dim() == 4:
            b, c, h, w = inputs.shape
            # For images, reshape transform_params from Bx(C*?)xHxW to BxCxHxWx?
            transform_params = transform_params.reshape(b, c, -1, h, w).permute(
                0, 1, 3, 4, 2
            )
        elif inputs.dim() == 2:
            b, d = inputs.shape
            # For 2D data, reshape transform_params from Bx(D*?) to BxDx?
            transform_params = transform_params.reshape(b, d, -1)

        outputs, logabsdet = self._piecewise_cdf(inputs, transform_params, inverse)

        return outputs, utils.sum_except_batch(logabsdet)

    def _piecewise_cdf(self, inputs, transform_params, inverse=False):
        raise NotImplementedError()


class PiecewiseRationalQuadraticCoupling(PiecewiseCoupling):
    def __init__(
        self,
        mask,
        transform_net_create_fn,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        apply_unconditional_transform=False,
        img_shape=None,
        min_bin_width=splines.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=splines.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=splines.DEFAULT_MIN_DERIVATIVE,
        num_cond_channels=0,
    ):
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        # Split tails parameter if needed
        features_vector = torch.arange(len(mask))
        identity_features = features_vector.masked_select(mask <= 0)
        transform_features = features_vector.masked_select(mask > 0)
        if isinstance(tails, list) or isinstance(tails, tuple):
            self.tails = [tails[i] for i in transform_features]
            tails_ = [tails[i] for i in identity_features]
        else:
            self.tails = tails
            tails_ = tails

        if torch.is_tensor(tail_bound):
            tail_bound_ = tail_bound[identity_features]
        else:
            self.tail_bound = tail_bound
            tail_bound_ = tail_bound

        if apply_unconditional_transform:
            unconditional_transform = lambda features: PiecewiseRationalQuadraticCDF(
                shape=[features] + (img_shape if img_shape else []),
                num_bins=num_bins,
                tails=tails_,
                tail_bound=tail_bound_,
                min_bin_width=min_bin_width,
                min_bin_height=min_bin_height,
                min_derivative=min_derivative,
            )
        else:
            unconditional_transform = None

        super().__init__(
            mask,
            transform_net_create_fn,
            unconditional_transform=unconditional_transform,
            num_cond_channels=num_cond_channels,
        )

        if torch.is_tensor(tail_bound):
            self.register_buffer("tail_bound", tail_bound[transform_features])

    def _transform_dim_multiplier(self):
        if self.tails == "linear":
            return self.num_bins * 3 - 1
        elif self.tails == "circular":
            return self.num_bins * 3
        else:
            return self.num_bins * 3 + 1

    def _piecewise_cdf(self, inputs, transform_params, inverse=False):
        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        if hasattr(self.transform_net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.transform_net.hidden_features)
        elif hasattr(self.transform_net, "hidden_channels"):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_channels)
            unnormalized_heights /= np.sqrt(self.transform_net.hidden_channels)
        else:
            warnings.warn(
                "Inputs to the softmax are not scaled down: initialization might be bad."
            )

        if self.tails is None:
            spline_fn = splines.rational_quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = splines.unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        return spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )


class CircularCoupledRationalQuadraticSpline(Flow):
    """
    Neural spline flow coupling layer with circular coordinates
    """

    def __init__(
        self,
        num_input_channels,
        num_blocks,
        num_hidden_channels,
        ind_circ,
        num_bins=8,
        tail_bound=3.0,
        activation=nn.ReLU,
        dropout_probability=0.0,
        reverse_mask=False,
        mask=None,
        init_identity=True,
        num_cond_channels=0,
        periodic_conditioning=False,
        NO_frequencies=1,
        use_fab_periodic_conditioning=True,
        use_cos_sin_periodic_representation=True,
    ):
        """Constructor

        Args:
          num_input_channels (int): Flow dimension
          num_blocks (int): Number of residual blocks of the parameter NN
          num_hidden_channels (int): Number of hidden units of the NN
          ind_circ (Iterable): Indices of the circular coordinates
          num_bins (int): Number of bins
          tail_bound (float or Iterable): Bound of the spline tails
          activation (torch module): Activation function
          dropout_probability (float): Dropout probability of the NN
          reverse_mask (bool): Flag whether the reverse mask should be used
          mask (torch tensor): Mask to be used, alternating masked generated is None
          init_identity (bool): Flag, initialize transform as identity
          num_cond_channels (int): Number of conditional channels
          periodic_conditioning (bool): Flag whether to use periodic conditioning representation.
          NO_frequencies (int): Number of frequencies to use for periodic conditioning.
          use_fab_periodic_conditioning (bool): Flag whether to use fab periodic conditioning or our implementation.
          use_cos_sin_periodic_representation (bool): Flag whether to use cos/sin representation for periodic representation of coupling layers parameter calculation.
        """
        super().__init__()

        self.periodic_conditioning = periodic_conditioning
        self.use_fab_periodic_conditioning = use_fab_periodic_conditioning
        self.NO_frequencies = NO_frequencies

        if periodic_conditioning and not use_fab_periodic_conditioning:
            num_cond_channels *= 2 * NO_frequencies

        if periodic_conditioning and use_fab_periodic_conditioning:
            self.periodic_conditioning_module = PeriodicFeaturesElementwise(
                2, [0, 1], scale=1.0
            )

        if mask is None:
            mask = create_alternating_binary_mask(num_input_channels, even=reverse_mask)
        features_vector = torch.arange(num_input_channels)
        identity_features = features_vector.masked_select(mask <= 0)

        ind_circ = torch.tensor(ind_circ)

        identity_circ_index = []
        identity_circ_index_original = []
        for i, id in enumerate(identity_features):
            if id in ind_circ:
                identity_circ_index += [i]
                identity_circ_index_original += [id.item()]

        if torch.is_tensor(tail_bound):
            # scale_pf = np.pi / tail_bound[ind_circ_id]
            scale_pf = (
                np.pi / tail_bound[identity_circ_index_original]
            )  # Fixed bug from original code
        else:
            scale_pf = np.pi / tail_bound

        def transform_net_create_fn(in_features, out_features, context_features=0):
            if len(identity_circ_index) > 0:
                if not use_cos_sin_periodic_representation:
                    pf = PeriodicFeaturesElementwise(
                        in_features, identity_circ_index, scale_pf
                    )
                else:
                    pf = PeriodicFeaturesCosSin(
                        in_features, identity_circ_index, scale_pf
                    )
            else:
                pf = None
            net = ResidualNet(
                in_features=in_features
                if not use_cos_sin_periodic_representation
                else in_features
                + len(
                    identity_circ_index
                ),  # circ features appear twice in cos/sin representation
                out_features=out_features,
                context_features=context_features if context_features > 0 else None,
                hidden_features=num_hidden_channels,
                num_blocks=num_blocks,
                activation=activation(),
                dropout_probability=dropout_probability,
                use_batch_norm=False,
                preprocessing=pf,
            )
            if init_identity:
                torch.nn.init.constant_(net.final_layer.weight, 0.0)
                torch.nn.init.constant_(
                    net.final_layer.bias, np.log(np.exp(1 - DEFAULT_MIN_DERIVATIVE) - 1)
                )
            return net

        tails = [
            "circular" if i in ind_circ else "linear" for i in range(num_input_channels)
        ]

        self.prqct = PiecewiseRationalQuadraticCoupling(
            mask=mask,
            transform_net_create_fn=transform_net_create_fn,
            num_bins=num_bins,
            tails=tails,
            tail_bound=tail_bound,
            apply_unconditional_transform=True,
            num_cond_channels=num_cond_channels,
        )

    def forward(self, z, context=None):  # from z to x
        if self.periodic_conditioning and not self.use_fab_periodic_conditioning:
            if self.NO_frequencies == 1:
                context = torch.cat(
                    # [torch.cos(context), torch.sin(context)],
                    [
                        torch.cos(context[:, 0:1]),  # + 137.0 / 180.0 * np.pi),
                        torch.sin(context[:, 0:1]),  # + 137.0 / 180.0 * np.pi),
                        torch.cos(context[:, 1:2]),
                        torch.sin(context[:, 1:2]),
                    ],
                    dim=-1
                    # [
                    #    torch.cos(context + 137.0 / 180.0 * np.pi),
                    #    torch.sin(context + 137.0 / 180.0 * np.pi),
                    # ],
                    # dim=-1,
                )

            else:  # general, one pair of cos/sin for each frequency
                to_cat = []
                for i in range(self.NO_frequencies):
                    to_cat.append(torch.cos(context[:, 0:1] * (i + 1)))
                    to_cat.append(torch.sin(context[:, 0:1] * (i + 1)))
                    to_cat.append(torch.cos(context[:, 1:2] * (i + 1)))
                    to_cat.append(torch.sin(context[:, 1:2] * (i + 1)))
                context = torch.cat(to_cat, dim=-1)

        elif self.periodic_conditioning and self.use_fab_periodic_conditioning:
            context = self.periodic_conditioning_module(context)

        z, log_det = self.prqct.inverse(z, context)
        return z, log_det.view(-1)

    def inverse(self, z, context=None):  # from x to z
        if self.periodic_conditioning and not self.use_fab_periodic_conditioning:
            if self.NO_frequencies == 1:
                context = torch.cat(
                    [
                        torch.cos(context[:, 0:1]),  # + 137.0 / 180.0 * np.pi),
                        torch.sin(context[:, 0:1]),  # + 137.0 / 180.0 * np.pi),
                        torch.cos(context[:, 1:2]),
                        torch.sin(context[:, 1:2]),
                    ],
                    dim=-1
                    # [
                    #    torch.cos(context + 137.0 / 180.0 * np.pi),
                    #    torch.sin(context + 137.0 / 180.0 * np.pi),
                    # ],
                    # dim=-1,
                )
            else:  # general, one pair of cos/sin for each frequency
                to_cat = []
                for i in range(self.NO_frequencies):
                    to_cat.append(torch.cos(context[:, 0:1] * (i + 1)))
                    to_cat.append(torch.sin(context[:, 0:1] * (i + 1)))
                    to_cat.append(torch.cos(context[:, 1:2] * (i + 1)))
                    to_cat.append(torch.sin(context[:, 1:2] * (i + 1)))
                context = torch.cat(to_cat, dim=-1)

        elif self.periodic_conditioning and self.use_fab_periodic_conditioning:
            context = self.periodic_conditioning_module(context)

        z, log_det = self.prqct(z, context)
        return z, log_det.view(-1)
