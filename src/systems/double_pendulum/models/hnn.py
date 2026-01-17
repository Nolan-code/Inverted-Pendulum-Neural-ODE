class MLP_T(nn.Module):
    """Kinetic energy network"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


class MLP_V(nn.Module):
    """Potential energy network"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

class HNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = MLP_T()
        self.V = MLP_V()

        # canonical symplectic matrix (θ1, θ2, ω1, ω2)
        self.register_buffer(
            "J",
            torch.tensor([
                [0., 0.,  1., 0.],
                [0., 0.,  0., 1.],
                [-1., 0., 0., 0.],
                [0., -1., 0., 0.]
            ])
        )

    def hamiltonian(self, X):
        sin1, cos1, sin2, cos2, w1, w2 = torch.split(X, 1, dim=1)

        T = self.T(torch.cat([sin1, cos1, sin2, cos2, w1, w2], dim=1))
        V = self.V(torch.cat([sin1, cos1, sin2, cos2], dim=1))

        return T + V

    def forward(self, X):
        X = X.detach().requires_grad_(True)

        H = self.hamiltonian(X)
        gradH = torch.autograd.grad(
            H.sum(),
            X,
            create_graph=True
        )[0]

        sin1, cos1, sin2, cos2 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

        dH_dtheta1 = gradH[:, 0] * cos1 - gradH[:, 1] * sin1
        dH_dtheta2 = gradH[:, 2] * cos2 - gradH[:, 3] * sin2

        dH_dw1 = gradH[:, 4]
        dH_dw2 = gradH[:, 5]

        gradH_canonical = torch.stack(
            [dH_dtheta1, dH_dtheta2, dH_dw1, dH_dw2],
            dim=1
        )

        dx = gradH_canonical @ self.J.T
        return dx
