import torch
import math


def call_model_batched(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    cond_tensor: torch.Tensor = None,
    pass_cond_tensor_as_tuple: bool = True,
    device: str = "cuda",
    batch_size: int = 128,
    move_back_to_cpu: bool = True,
    do_detach: bool = True,
    droplast: bool = False,
) -> torch.Tensor:
    """Call model batchwise.

    Args:
        model (torch.nn.Module): Model to call.
        input_tensor (torch.Tensor): Input tensor to split into batches.
        cond_tensor (torch.Tensor, optional): Conditional tensor to split into batches. Defaults to None.
        pass_cond_tensor_as_tuple: bool, optional): Pass the conditional tensor as a tuple. Defaults to True.
        device (str, optional): Device to copy each batch to. Defaults to "cuda".
        batch_size (int, optional): Size of one batch. Defaults to 128.
        move_back_to_cpu (bool, optional): Move the output back to the CPU. Defaults to True.
        do_detach (bool, optional): Detach the output tensor. Defaults to True.
        droplast (bool, optional): Drop the last batch if it is smaller than the batch size. Defaults to False.

    Returns:
        torch.Tensor: Stacked results.
    """

    if not droplast:
        total_NO_batches = int(math.ceil(input_tensor.shape[0] / batch_size))
    else:
        total_NO_batches = int(input_tensor.shape[0] / batch_size)
    NO_samples_to_process = (
        input_tensor.shape[0] if not droplast else total_NO_batches * batch_size
    )

    output_tensors = None

    for i in range(total_NO_batches):
        if cond_tensor is None:
            batch_result = model(
                input_tensor[i * batch_size : (i + 1) * batch_size].to(device),
            )
        else:
            batch_result = model(
                input_tensor[i * batch_size : (i + 1) * batch_size].to(device),
                (cond_tensor[i * batch_size : (i + 1) * batch_size].to(device),)
                if pass_cond_tensor_as_tuple
                else cond_tensor[i * batch_size : (i + 1) * batch_size].to(device),
            )

        if output_tensors is None:
            if (
                type(batch_result) is tuple
            ):  # Handle multiple output models (tuple output)
                output_tensors = []
                for result in batch_result:
                    output_tensors.append(
                        torch.empty((NO_samples_to_process, *result.shape[1:]))
                    )
            else:
                output_tensors = torch.empty(
                    (NO_samples_to_process, *batch_result.shape[1:]),
                    device=device if not move_back_to_cpu else "cpu",
                )  # Only a single tensor

        if type(batch_result) is tuple:  # Handle multiple output models (tuple output)
            for j, result in enumerate(batch_result):
                if do_detach:
                    result = result.detach()
                if move_back_to_cpu:
                    result = result.cpu()

                output_tensors[j][i * batch_size : (i + 1) * batch_size] = result
        else:
            if do_detach:
                result = batch_result.detach()
            else:
                result = batch_result
            if move_back_to_cpu:
                result = result.cpu()

            output_tensors[i * batch_size : (i + 1) * batch_size] = result

    return output_tensors
