def hamiltonian_dynamics(model, x):
  x = x.clone().detach().requires_grad_(True)

  H = model(x).sum()

  grad = torch.autograd.grad(
      H,
      x,
      create_graph=True
  )[0]

  d_theta = grad[:,1]
  d_omega = -grad[:,0]

  return torch.stack([d_theta, d_omega], dim=1)
