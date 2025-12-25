## Results of the MLP
### First model
In this first experiment, the MLP directly predicts the next state of the system given the current state and the applied control force \( u \).

#### One-step prediction
- The MLP achieves low error in the one-step prediction task.
- Both training and test losses converge quiclky toward 1e-3, 1e-4.

#### Long-term rollout
- The learned dynamics reproduces the qualitative behavior of the trajectory.
- However, the amplitude and frequency of the trajectory is diverging from the true dynamics over time.
![Mecganical energy conservation](figures/MLP_vs_true.png)

#### Energy analysis
- The true system conserves mechanical energy.
- The MLP mechanical energy tend to diverge over time.

#### Phase portrait
- The phase-space structure is approximately captured.
- The long-term trajectories diverge from the expected dynamics due to error accumulation.
- The lack of closed trajectories in the learned phase portrait is a consequence of the model not conserving the mechanical energy.
![Mecganical energy conservation](figures/phase.png)
