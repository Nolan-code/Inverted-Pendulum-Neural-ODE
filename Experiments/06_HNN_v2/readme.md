### Fifth model
In this experiment, a Hamiltonian Neural Network is trained to learn the Hamiltonian of an inverted pendulum system.
The state of the system is represented using the tuple:

  x=(sin⁡θ,cos⁡θ,ω)

This representation is used to handle the periodicity of the angular variable θ.
The neural network learns a scalar-valued Hamiltonian H(x), from which the time derivatives of the state are recovered using Hamilton’s equations.
#### Analysis
- Despite trying a new architecture, the neural network still exhibits the same issue: it systematically places the stable equilibrium point at 0 instead of pi.
- The learned Hamiltonian is defined up to a symmetry in the angular coordinate. While the model reproduces the correct dynamics, the absolute position of the stable equilibrium is not identifiable from data alone.

#### Conclusion
- This final HNN experiment demonstrates that:
  - Hamiltonian Neural Networks can successfully learn locally accurate conservative dynamics
  - However, without additional global constraints, they may fail to recover:
    - The correct energy landscape
    - The true location of stable equilibrium point
- These limitations motivate the transition to another model,a Langrangian neural network where:
  - The potential energy is directly learned
  - Stable equilibria correspond to minima of a learned scalar function
  - The physical structure of the system is more strongly constrained
