import torch
import wandb
from main.active_learning.active_learning_dataset import ActiveLearningDataset
from torch.autograd import Function
from typing import Tuple
from main.models.flow_base import ConditionalFlowBase
from main.systems.system import System
from main.utils.call_model_batched import call_model_batched
import traceback
import logging
from main.systems.aldp.coord_trafo import alternative_filter_chirality
import numpy as np


class FunctionWrapper(Function):
    """This is a function wrapper that can be used as a "pseudo"
    pytorch function for functions where the output and gradient are
    already known and stored somewhere. We use this for the reweighting
    strategy, where we reuse stored previous potential energy evaluations.
    """

    @staticmethod
    def forward(ctx, input, value, grad):
        ctx.save_for_backward(input, grad)
        return value

    @staticmethod
    def backward(ctx, grad_output):
        input, grad = ctx.saved_tensors
        return (
            grad_output * grad,
            None,
            None,
        )  # None for the gradients of function and grad


def train_single_batch_by_example(
    x: torch.Tensor,
    cond_flow: ConditionalFlowBase,
    system: System,
) -> Tuple[torch.Tensor, float]:
    """Train the model by example.

    Args:
        coords (torch.Tensor): Coordinates.
        cond_flow (ConditionalFlowBase): Conditional flow.
        system (System): The target system.

    Returns:
        torch.Tensor: KL loss Tensor
        float: KL loss scalar
    """

    x_int = system.cartesian_to_internal(x)
    x_int_fg = x_int[:, system.FG_mask]
    x_int_cg = x_int[:, system.CG_mask]

    kld = cond_flow.forward_kld(x_int_fg, context=x_int_cg)

    return kld, kld.item()


def train_epoch_reweighted_by_probability(
    cond_flow: ConditionalFlowBase,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    AL_dataset: ActiveLearningDataset,
    system: System,
    gradient_clipping_value: float = None,
) -> int:
    """Train the model by probability using the reweighting approach.
    This runs a full epoch, not just a single batch.

    Args:
        cond_flow (ConditionalFlowBase): Conditional flow.
        optimizer (torch.optim.Optimizer): Optimizer.
        epoch (int): Current epoch.
        AL_dataset (ActiveLearningDataset): Dataset for active learning.
        system (System): The target system.
        gradient_clipping_value (float, optional): Gradient clipping value. Defaults to None.

    Returns:
        int: Number of evaluations of target system's potential energy function in this epoch.
    """

    sum_KL_loss_by_probability = 0
    sum_sum_of_weights_vector = 0
    sum_KL_loss_by_probability_unnormalized = 0
    sum_median_weight_vector = 0
    sum_grad_norm = 0
    sum_R_chirality = 0
    max_R_chirality = 0
    sum_R_chirality_alternative = 0
    max_R_chirality_alternative = 0
    sum_R_chirality_alternative_1 = 0
    max_R_chirality_alternative_1 = 0
    sum_R_chirality_combined = 0
    max_R_chirality_combined = 0
    NO_evaluations = 0
    NO_evaluations_after_filtering = 0

    highest_loss_values_clipped = []
    highest_loss_values_kept = []

    batch_idx = 0

    latest_exception = None
    exception_counter = 0

    AL_dataset.shuffle()

    for (
        current_target_log_prob,
        current_grads,
        current_xs_int,
        current_s,
        current_z0s,
        current_log_pxgiveny_0,
        need_evaluation,  # Those have just been sampled or are getting reevaluated
        current_evaluation_counters,
    ) in AL_dataset:
        batch_idx += 1
        optimizer.zero_grad()

        (
            current_s,
            current_z0s,
        ) = (
            current_s.to("cuda"),
            current_z0s.to("cuda"),
        )

        NO_evaluations += torch.sum(need_evaluation).item()

        if wandb.config.resampling_probability == 1.0:
            assert (
                torch.sum(need_evaluation) == need_evaluation.shape[0]
            ), "If resampling_probability is 1.0, then all points should be resampled!"

        # region Determine the weights of the samples in this batch (only the ones that require reweighting)

        weights = torch.ones(current_z0s.shape[0], device="cuda")

        current_z1s = current_z0s.clone()

        need_reweighting = torch.logical_not(
            need_evaluation
        )  # those have not just been sampled but have been sampled in a previous step => reweighting is needed

        xs_int_need_reweighting = current_xs_int[need_reweighting].to("cuda")
        s_need_reweighting = current_s[need_reweighting]

        # Pass already-assigned xs through the model to get z_1 and p_1 for reweighting:

        xs_int_fg_need_reweighting = xs_int_need_reweighting[:, system.FG_mask]
        with torch.no_grad():  # we do not need a gradient here, this is just the reweighting
            if xs_int_fg_need_reweighting.shape[0] > 0:
                if wandb.config.resampling_probability == 1.0:
                    raise Exception(
                        "If resampling_probability is 1.0, then there should be no reweighting!"
                    )

                try:
                    zs = cond_flow.inverse(
                        xs_int_fg_need_reweighting, context=s_need_reweighting
                    )
                except Exception as ex:
                    latest_exception = (
                        "Skipping this batch and continuing with the next one.\n"
                        + traceback.format_exc()
                    )
                    exception_counter += 1
                    continue
            else:
                zs = torch.empty(
                    (0, xs_int_fg_need_reweighting.shape[1]), device="cuda"
                )

        before_exp = cond_flow.q0.log_prob(zs) - cond_flow.q0.log_prob(
            current_z0s[need_reweighting]
        )
        before_exp -= torch.median(before_exp)

        weights[need_reweighting] = torch.exp(before_exp)

        current_z1s[need_reweighting] = zs

        # Just to make sure that there are no gradients
        current_z1s = current_z1s.detach()
        weights = weights.detach()

        # Increase the weight of the points that have been reevaluated
        weights[need_evaluation] = (
            weights[need_evaluation] * wandb.config.multiplier_for_reevaluation
        )

        sum_sum_of_weights_vector += torch.sum(weights)

        # endregion

        if wandb.config.target_system["name"] == "aldp":
            # Throw an error if current_s not in range -pi..pi
            if torch.any(current_s < -torch.pi) or torch.any(current_s > torch.pi):
                raise Exception(
                    "Some of the current_s values are not in the range -pi..pi. This should not happen!"
                )

        # region Make the actual backward pass

        # TODO: This could be sped up by using the jacobian of the previous step for the ones that needed reweighting!
        try:
            x_int_fg, jac = cond_flow.forward_and_log_det(
                current_z1s, context=current_s
            )
        except Exception as ex:
            latest_exception = (
                "Skipping this batch and continuing with the next one.\n"
                + traceback.format_exc()
            )
            exception_counter += 1
            continue

        x_int = torch.empty(
            (current_z1s.shape[0], cond_flow.q0.ndim + current_s.shape[1]),
            device="cuda",
        )
        x_int[:, system.FG_mask] = x_int_fg
        x_int[:, system.CG_mask] = current_s

        ##### Chirality filtering #####

        if wandb.config.flow_architecture["filter_chirality"]:
            L_mask = system.filter_chirality(
                x_int,
                threshold=wandb.config.flow_architecture["fab_filtering_threshold"],
            ).cpu()
            L_mask_alternative = alternative_filter_chirality(x_int, system.trafo).cpu()
            L_mask_combined = L_mask & L_mask_alternative

            L_mask_alternative_1 = alternative_filter_chirality(
                x_int, system.trafo, use_hydrogen_carbon_vector=True
            ).cpu()

            # For logging:
            R_mask_sum = torch.sum(~L_mask).cpu().item()
            R_mask_sum_alternative = torch.sum(~L_mask_alternative).cpu().item()
            R_mask_sum_alternative_1 = torch.sum(~L_mask_alternative_1).cpu().item()
            R_mask_sum_combined = torch.sum(~L_mask_combined).cpu().item()

            sum_R_chirality += R_mask_sum
            sum_R_chirality_alternative += R_mask_sum_alternative
            sum_R_chirality_alternative_1 += R_mask_sum_alternative_1
            sum_R_chirality_combined += R_mask_sum_combined

            if R_mask_sum > max_R_chirality:
                max_R_chirality = R_mask_sum
            if R_mask_sum_alternative > max_R_chirality_alternative:
                max_R_chirality_alternative = R_mask_sum_alternative
            if R_mask_sum_alternative_1 > max_R_chirality_alternative_1:
                max_R_chirality_alternative_1 = R_mask_sum_alternative_1
            if R_mask_sum_combined > max_R_chirality_combined:
                max_R_chirality_combined = R_mask_sum_combined

            if wandb.config.flow_architecture["filtering_type"] == "alternative":
                L_mask = L_mask_alternative
            elif wandb.config.flow_architecture["filtering_type"] == "alternative_1":
                L_mask = L_mask_alternative_1
            elif wandb.config.flow_architecture["filtering_type"] == "combined":
                L_mask = L_mask_combined
            elif (
                wandb.config.flow_architecture["filtering_type"] == "none"
                or wandb.config.flow_architecture["filtering_type"] == "mirror"
            ):
                L_mask = torch.ones(x_int.shape[0], dtype=torch.bool, device="cpu")

            if torch.mean(1.0 * L_mask) > 0.1 or (
                not wandb.config.flow_architecture["dont_filter_below_10_percent"]
            ):
                current_evaluation_counters[
                    need_evaluation & (~L_mask)
                ] -= 1  # Didn't actually get evaluated, so undo the increment
            else:  # if not too much, then ignore
                L_mask = torch.ones(x_int.shape[0], dtype=torch.bool, device="cpu")

        else:
            L_mask = torch.ones(x_int.shape[0], dtype=torch.bool, device="cpu")

        NO_evaluations_after_filtering += torch.sum(need_evaluation & L_mask).item()

        # Freshly evaluated, store for reweighting / reusage later
        current_xs_int[need_evaluation & L_mask] = (
            x_int[need_evaluation & L_mask].detach().to("cpu")
        )

        # endregion

        # region Make the needed evaluations of the target potential energy function

        if torch.sum((need_evaluation & L_mask)) > 0:
            log_prob, log_prob_grad = system.target_log_prob_and_grad(
                current_xs_int[need_evaluation & L_mask]
            )
            log_prob, log_prob_grad = (
                log_prob.detach(),
                log_prob_grad.detach(),
            )

            # Store them for later reuse (this also modified the original tensors self.target_log_prob and self.grads in the AL dataset)
            current_target_log_prob[need_evaluation & L_mask] = log_prob[:, None]
            current_grads[need_evaluation & L_mask] = log_prob_grad

            # if torch.sum((~need_evaluation) & (~L_mask)) > 0:
            #    raise Exception(
            #        "There are points that do not need to be evaluated (which means that they have been successfully evaluated before)\n"
            #        + "but they are not in the L-form. This should not happen!"
            #    )

        # endregion

        current_target_log_prob = current_target_log_prob.detach().to("cuda")
        current_grads = current_grads.detach().to("cuda")

        function_wrapper = FunctionWrapper.apply
        log_probability_x = function_wrapper(
            x_int[L_mask], current_target_log_prob[L_mask], current_grads[L_mask]
        )[:, 0]

        # We need this to compute the free energy:
        current_log_pxgiveny_0[need_evaluation & L_mask] = (
            (
                cond_flow.q0.log_prob(current_z0s[need_evaluation & L_mask])[:, None]
                - jac[need_evaluation & L_mask].detach()[:, None]  # Watch the sign!
            )
            .detach()
            .cpu()
        )

        # Normalize the weights to sum to len(weights) => make the training more stable
        if wandb.config.normalize_weight_vector:
            # For logging, calculate the KL loss by probability with unnormalized weights first
            KL_loss_by_probability_scalar_unnormalized = (
                -log_probability_x - jac[L_mask]
            ) * weights[
                L_mask
            ]  # with unnormalized weights!

            if wandb.config.flow_architecture.get("skip_top_k_losses") is not None:
                # Skip the top k losses
                k = wandb.config.flow_architecture["skip_top_k_losses"]

                if k > 0:
                    sorted_losses, _ = torch.sort(
                        KL_loss_by_probability_scalar_unnormalized, descending=True
                    )
                    KL_loss_by_probability_scalar_unnormalized = sorted_losses[k:]

                    highest_loss_values_clipped.append(
                        sorted_losses[0].detach().cpu().item()
                    )
                    highest_loss_values_kept.append(
                        sorted_losses[k].detach().cpu().item()
                    )

            sum_KL_loss_by_probability_unnormalized += (
                KL_loss_by_probability_scalar_unnormalized.mean().detach().cpu().item()
            )

            # Now normalize the weights:
            weights[L_mask] = (
                weights[L_mask] / torch.sum(weights[L_mask]) * len(weights[L_mask])
            )

        else:
            raise Exception("Not reweighting the weights is not supported anymore.")

        # Log the median of the normalized weights vector
        sum_median_weight_vector += torch.median(weights[L_mask].detach()).cpu().item()

        if (
            wandb.config.resampling_probability == 1.0
            and wandb.config.multiplier_for_reevaluation == 1.0
        ):
            # All weights should be 1.0
            assert torch.allclose(weights[L_mask], torch.tensor(1.0))

        KL_loss_by_probability = (-log_probability_x - jac[L_mask]) * weights[
            L_mask
        ]  # weighted!

        if wandb.config.flow_architecture.get("skip_top_k_losses") is not None:
            # Skip the top k losses
            k = wandb.config.flow_architecture["skip_top_k_losses"]

            if k > 0:
                sorted_losses, _ = torch.sort(KL_loss_by_probability, descending=True)
                KL_loss_by_probability = sorted_losses[k:]

        KL_loss_by_probability = KL_loss_by_probability.mean()

        KL_loss_by_probability_scalar = KL_loss_by_probability.cpu().item()
        l = KL_loss_by_probability

        sum_KL_loss_by_probability += KL_loss_by_probability_scalar

        # Check for nan or inf
        if torch.isnan(l) or torch.isinf(l):
            logging.info(f"Loss while training cond flow is nan or inf (epoch {epoch})")
            continue

        l.backward()

        if gradient_clipping_value is not None:
            norm = torch.nn.utils.clip_grad_norm_(
                cond_flow.parameters(), gradient_clipping_value
            )
        else:
            # Just for logging
            grads = [p.grad for p in cond_flow.parameters() if p.grad is not None]
            norm = torch.norm(
                torch.stack([torch.norm(g.detach(), 2.0).to("cuda") for g in grads]),
                2.0,
            )

        sum_grad_norm += norm

        optimizer.step()

    if latest_exception is not None:
        logging.info(
            f"Exceptions ({exception_counter} times) occurred while training by probability (epoch {epoch}). Latest one:\n"
            + latest_exception
        )

        # If more than 50% of the batches failed, we should probably stop training
        if exception_counter > (len(AL_dataset) / 2):
            raise Exception(
                f"Too many exceptions ({exception_counter} of {len(AL_dataset)} batches) occurred while training by probability (epoch {epoch})."
            )

    wandb.log(
        {
            "median_weight_vector_after_normalization": sum_median_weight_vector
            / batch_idx,
            "grad_norm": sum_grad_norm / batch_idx,
            "KL_loss_by_probability_unnormalized": sum_KL_loss_by_probability_unnormalized
            / batch_idx,
            "total_loss": sum_KL_loss_by_probability / batch_idx,
            "KL_loss_by_example": 0.0,
            "KL_loss_by_probability": sum_KL_loss_by_probability / batch_idx,
            "sum_of_weights_vector_before_normalization": sum_sum_of_weights_vector
            / batch_idx,
            "R_chirality": sum_R_chirality / batch_idx,
            "R_chirality_alternative": sum_R_chirality_alternative / batch_idx,
            "R_chirality_alternative_1": sum_R_chirality_alternative_1 / batch_idx,
            "R_chirality_combined": sum_R_chirality_combined / batch_idx,
            "NO_evaluations_per_epoch": NO_evaluations,
            "NO_evaluations_after_filtering_per_epoch": NO_evaluations_after_filtering,
            "learning_rate_cond_flow": optimizer.param_groups[0]["lr"],
            "max_R_chirality": max_R_chirality,
            "max_R_chirality_alternative": max_R_chirality_alternative,
            "max_R_chirality_alternative_1": max_R_chirality_alternative_1,
            "max_R_chirality_combined": max_R_chirality_combined,
            "highest_loss_values_clipped_mean": (
                np.mean(highest_loss_values_clipped)
                if len(highest_loss_values_clipped) > 0
                else 0.0
            ),
            "highest_loss_values_kept_mean": (
                np.mean(highest_loss_values_kept)
                if len(highest_loss_values_kept) > 0
                else 0.0
            ),
            "highest_loss_values_clipped_max": (
                np.max(highest_loss_values_clipped)
                if len(highest_loss_values_clipped) > 0
                else 0.0
            ),
            "highest_loss_values_kept_max": (
                np.max(highest_loss_values_kept)
                if len(highest_loss_values_kept) > 0
                else 0.0
            ),
        },
        step=epoch,
    )

    return NO_evaluations, NO_evaluations_after_filtering


def calculate_test_KL_by_probability_loss(
    cond_flow: ConditionalFlowBase,
    AL_dataset: ActiveLearningDataset,
    system: System,
) -> float:
    """Calculate the test KL loss. This has to be implemented separately, because
    it doesn't use any reweighting.

    Args:
        cond_flow (ConditionalFlowBase): Conditional flow.
        AL_dataset (ActiveLearningDataset): AL dataset.
        system (System): The target system.

    Returns:
        float: KL loss by probability.
    """

    s = AL_dataset.get_current_s_test()

    if wandb.config.target_system["name"] == "mb":
        # Make sure that we have enough samples for the test loss calculation
        # Repeat s 10 times
        s = s.repeat(10, 1)

    counter = 0
    while True:
        try:
            # Prepare zs:
            zs = cond_flow.q0.sample(s.shape[0])

            helper_fn = lambda z, samples: cond_flow.forward_and_log_det(
                z, context=samples
            )

            with torch.no_grad():
                # Returns a tuple of (samples_backmapped [N,1], jacobian [N])
                samples_backmapped = call_model_batched(
                    helper_fn,
                    zs,
                    cond_tensor=s,
                    device="cuda",
                    batch_size=wandb.config.batch_size_probability,
                    droplast=True
                    if wandb.config.flow_architecture.get("skip_top_k_losses")
                    is not None
                    else False,
                    do_detach=False,
                    pass_cond_tensor_as_tuple=False,
                )
        except Exception as ex:
            logging.info(
                "Exception while testing by probability:\n" + traceback.format_exc()
            )
            counter += 1
            if counter > 3:
                logging.info(
                    "Too many exceptions while testing by probability!\n"
                    + "We will just skip the test loss calculation for this epoch.\n"
                )
                return None
            continue
        break

    x_int_fg = samples_backmapped[0]
    jac = samples_backmapped[1]

    x_int = torch.empty((x_int_fg.shape[0], cond_flow.q0.ndim + s.shape[1]))
    x_int[:, system.FG_mask] = x_int_fg
    x_int[:, system.CG_mask] = s[: x_int_fg.shape[0]]

    log_probability_x = system.target_log_prob(x_int)

    KL_loss_by_probability = -log_probability_x - jac  # not weighted!

    k = wandb.config.flow_architecture.get("skip_top_k_losses")
    if k is not None and k > 0:
        # Apply the k-skipping within each batch that was sampled individually
        # to have exactly the same setup as during training

        final_loss_values = torch.empty(
            (
                int(
                    KL_loss_by_probability.shape[0]
                    - (
                        KL_loss_by_probability.shape[0]
                        / wandb.config.batch_size_probability
                    )
                    * k
                ),
            ),
        )

        j = 0
        for i in range(
            0, KL_loss_by_probability.shape[0], wandb.config.batch_size_probability
        ):
            sorted_losses, _ = torch.sort(
                KL_loss_by_probability[i : i + wandb.config.batch_size_probability],
                descending=True,
            )
            final_loss_values[
                j : j + (wandb.config.batch_size_probability - k)
            ] = sorted_losses[k:]

            j += wandb.config.batch_size_probability - k

        KL_loss_by_probability_scalar = final_loss_values.mean().item()

    else:
        KL_loss_by_probability_scalar = KL_loss_by_probability.mean().item()

    return KL_loss_by_probability_scalar
