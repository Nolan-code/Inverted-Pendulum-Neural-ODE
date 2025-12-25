def dH_dt(model, x):   # used to check if the HNN is hamiltonian (the output need to be close to 0)
    x = x.clone().detach().requires_grad_(True)
    dx = hamiltonian_dynamics_norm(model, x)

    H = model((x - X_mean) / X_std).sum()
    grad_H = torch.autograd.grad(H, x)[0]

    return torch.dot(grad_H, dx)
