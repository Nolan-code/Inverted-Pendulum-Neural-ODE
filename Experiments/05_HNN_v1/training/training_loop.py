#-------------
# Load data
#-------------

data = np.load("4.5_HNN_dataset.npz")
split = np.load("4.6_splits_HNN.npz")

X = np.hstack(data["X"])
X = X.reshape(15300, 2)
Y = data["Y"]
Y = Y.reshape(15300, 2)
print(X.shape)

#---------------
# Normalization
#---------------

"""X_mean = X.mean()
X_std  = X.std()

Y_mean = Y.mean()
Y_std  = Y.std()

X = (X - X_mean) / X_std
Y = (Y - Y_mean) / Y_std"""


X_train = X[split["train_idx"]]
Y_train = Y[split["train_idx"]]

X_test = X[split["test_idx"]]
Y_test = Y[split["test_idx"]]

#-------------------
# Convert to tensor
#-------------------

X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train, Y_train)
test_dataset = TensorDataset(X_test, Y_test)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

#---------
#  Model
#---------

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
λ_eq = 1

#----------------
# Training loop
#----------------

n_epochs = 350

for epoch in range(n_epochs):
  model.train()
  train_loss = 0.0

  for X_batch, Y_batch in train_loader:

    theta = X_batch[:, 0]
    omega = X_batch[:, 1]

    X_tilde = torch.stack([
        torch.sin(theta),
        torch.cos(theta),
        omega
    ], dim=1)

    dtheta_pred, domega_pred = hamiltonian_dynamics(
        model, X_tilde, theta
    )

    x_eq = torch.stack([
        torch.sin(torch.tensor([np.pi])),
        torch.cos(torch.tensor([np.pi])),
        torch.tensor([0.0])
    ], dim=1).requires_grad_(True)

    H_eq = model(x_eq).sum()

    grad_eq = torch.autograd.grad(
       H_eq,
       x_eq,
       create_graph=True
    )[0]

    loss_eq = torch.sum(grad_eq**2)  # to fix the stable equilibrium point at pi

    loss = (
        loss_fn(dtheta_pred, Y_batch[:,0]) +
        loss_fn(domega_pred, Y_batch[:,1]) +
        λ_eq * loss_eq
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    train_loss += loss.item()

  train_loss /= len(train_loader)


  model.eval()
  test_loss = 0.0

  for X_batch, Y_batch in test_loader:
      theta = X_batch[:,0]
      omega = X_batch[:,1]
      X_tilde = torch.stack([
        torch.sin(theta),
        torch.cos(theta),
        omega
      ], dim=1)

      dtheta_pred, domega_pred = hamiltonian_dynamics(
        model, X_tilde, theta
      )

      loss = (
        loss_fn(dtheta_pred, Y_batch[:,0]) +
        loss_fn(domega_pred, Y_batch[:,1]) 
      )
      test_loss += loss.item()

  test_loss /= len(test_loader)
  print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

#-------------
# Save model
#-------------

torch.save(model.state_dict(), "Upgrade_HNN.pth")
