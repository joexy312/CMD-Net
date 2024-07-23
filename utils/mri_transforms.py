import torch
from utils.fftc import fft2_2c, ifft2_2c
from typing import Dict, Optional, Sequence, Tuple, Union


def conjugate(data: torch.Tensor) -> torch.Tensor:

    data = data.clone()
    data[..., 1] = data[..., 1] * -1.0
    return data


def complex_multiply(x, y, u, v):
    """
    Computes (x+iy) * (u+iv) = (x * u - y * v) + (x * v + y * u)i = z1 + iz2
    Returns (real z1, imaginary z2)
    """

    z1 = x * u - y * v
    z2 = x * v + y * u

    return torch.stack((z1, z2), dim=-1)


def combine_all_coils(image, sensitivity):
    """return sensitivity combined images from all coils"""
    assert image.size(-1) == 2
    assert sensitivity.size(-1) == 2
    combined = complex_multiply(sensitivity[..., 0],
                                -sensitivity[..., 1],
                                image[..., 0],
                                image[..., 1])

    return combined.sum(dim=0)


def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.shape[-1] == 2
    return (data ** 2).sum(dim=-1).sqrt()


def complex_multiplication(input_tensor: torch.Tensor, other_tensor: torch.Tensor) -> torch.Tensor:

    complex_index = -1

    real_part = input_tensor[..., 0] * other_tensor[..., 0] - input_tensor[..., 1] * other_tensor[..., 1]
    imaginary_part = input_tensor[..., 0] * other_tensor[..., 1] + input_tensor[..., 1] * other_tensor[..., 0]

    multiplication = torch.cat(
        [
            real_part.unsqueeze(dim=complex_index),
            imaginary_part.unsqueeze(dim=complex_index),
        ],
        dim=complex_index,
    )

    return multiplication


def reduce_operator(
    coil_data: torch.Tensor,
    sensitivity_map: torch.Tensor,
    dim: int = 1,
) -> torch.Tensor:

    return complex_multiplication(conjugate(sensitivity_map), coil_data).sum(dim)


def expand_operator(
    data: torch.Tensor,
    sensitivity_map: torch.Tensor,
    dim: int = 1,
) -> torch.Tensor:

    return complex_multiplication(sensitivity_map, data.unsqueeze(dim))


def forward_operator(image, sampling_mask, sensitivity_map):  # PFS
    forward = torch.where(
        sampling_mask == 0,
        torch.tensor([0.0], dtype=image.dtype).to(image.device),
        fft2_2c(expand_operator(image, sensitivity_map, 1), dim=(2, 3)),
    )
    return forward


def backward_operator(kspace, sampling_mask, sensitivity_map):  # (PFS)^(-1)
    backward = reduce_operator(
        ifft2_2c(
            torch.where(
                sampling_mask == 0,
                torch.tensor([0.0], dtype=kspace.dtype).to(kspace.device),
                kspace,
            ),
            (2, 3)
        ),
        sensitivity_map,
        1,
    )
    return backward


def rss(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS).

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    return torch.sqrt((data ** 2).sum(dim))


def rss_complex(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS) for complex inputs.

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    return torch.sqrt(complex_abs_sq(data).sum(dim))


def complex_abs(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data: A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        Absolute value of data.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return ((data ** 2+1e-8).sum(dim=-1)).sqrt()


def complex_abs_sq(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the squared absolute value of a complex tensor.

    Args:
        data: A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        Squared absolute value of data.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data ** 2).sum(dim=-1)


def complex_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Complex multiplication.

    This multiplies two complex tensors assuming that they are both stored as
    real arrays with the last dimension being the complex dimension.

    Args:
        x: A PyTorch tensor with the last dimension of size 2.
        y: A PyTorch tensor with the last dimension of size 2.

    Returns:
        A PyTorch tensor with the last dimension of size 2.
    """
    if not x.shape[-1] == y.shape[-1] == 2:
        raise ValueError("Tensors do not have separate complex dim.")

    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]

    return torch.stack((re, im), dim=-1)


def complex_conj(x: torch.Tensor) -> torch.Tensor:
    """
    Complex conjugate.

    This applies the complex conjugate assuming that the input array has the
    last dimension as the complex dimension.

    Args:
        x: A PyTorch tensor with the last dimension of size 2.
        y: A PyTorch tensor with the last dimension of size 2.

    Returns:
        A PyTorch tensor with the last dimension of size 2.
    """
    if not x.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return torch.stack((x[..., 0], -x[..., 1]), dim=-1)


def center_crop_to_smallest(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a center crop on the larger image to the size of the smaller.

    The minimum is taken over dim=-1 and dim=-2. If x is smaller than y at
    dim=-1 and y is smaller than x at dim=-2, then the returned dimension will
    be a mixture of the two.

    Args:
        x: The first image.
        y: The second image.

    Returns:
        tuple of tensors x and y, each cropped to the minimim size.
    """
    smallest_width = min(x.shape[-1], y.shape[-1])
    smallest_height = min(x.shape[-2], y.shape[-2])
    x = center_crop(x, (smallest_height, smallest_width))
    y = center_crop(y, (smallest_height, smallest_width))

    return x, y


def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.

    Returns:
        The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]