import torch


def quantize_gradient_tensor(grad, num_bits=5):
    if grad is None or num_bits is None or num_bits <= 0:
        return grad

    grad_min = grad.min()
    grad_max = grad.max()
    if torch.isclose(grad_min, grad_max):
        return grad

    qmin = 0.0
    qmax = float(2 ** num_bits - 1)
    scale = (grad_max - grad_min) / (qmax - qmin)
    scale = torch.clamp(scale, min=1e-8)

    quantized = torch.round(torch.clamp((grad - grad_min) / scale + qmin, qmin, qmax))
    return (quantized - qmin) * scale + grad_min


def add_gradient_noise(grad, noise_std=0.1):
    if grad is None or noise_std <= 0:
        return grad

    grad_scale = grad.detach().std(unbiased=False)
    if torch.isnan(grad_scale) or torch.isclose(grad_scale, torch.tensor(0.0, device=grad.device, dtype=grad.dtype)):
        return grad

    noise = torch.randn_like(grad) * (grad_scale * noise_std)
    return grad + noise


def process_model_gradients(model, num_bits=5, noise_std=0.1):
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        parameter.grad.data = quantize_gradient_tensor(parameter.grad.data, num_bits=num_bits)
        parameter.grad.data = add_gradient_noise(parameter.grad.data, noise_std=noise_std)
