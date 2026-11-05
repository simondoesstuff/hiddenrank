# Rate network and analysis

The simulator implements Eq. 1–2 of Mastrogiuseppe & Ostojic (2018):

\[
\tau\dot x=-x+J\tanh(x)+I(t),\qquad
J=g\chi+MN^\mathsf{T}/N_\mathrm{neurons}.
\]

- `RateNetwork(m, n, random_matrix=None, g=0, tau=1)` takes `(neurons, rank)`
  factors. `random_matrix` is **already scaled** to variance `1/neurons`.
- The structured product costs O(neurons × rank). Dense random connectivity
  costs O(neurons²); `gaussian_network(..., g=0)` allocates no dense matrix.
- `gaussian_network(key, size, rank, ...)` draws independent Gaussian vector
  pairs with configurable means, standard deviations, and within-pair correlation.
  Sample moments fluctuate at finite population size.
- `simulate(network, initial_state, inputs, dt=0.1, method="rk4", sample_every=1)`
  uses JAX scans; inputs have shape `(steps, neurons)` and are constant per step.
  Euler is also available. Times and states include t=0 and the final state.
  Sampling retains only checkpoint states; the full input array remains resident.
- `Simulation.rates` is `tanh(states)`. Time is in units of `tau` when `tau=1`.

## Population geometry

- `overlaps(activity, vectors)` computes population inner products divided by N.
  The paper's recurrent overlap is `overlaps(tanh(states), network.n)`.
- `readout(states, weights)` computes `mean(weights * tanh(states))`.
- `project(activity, vectors)` returns least-squares coefficients in the columns
  of the supplied basis; raw overlaps do not generally recover these coefficients.
- `pca(activity, n_components)` centers observations and returns scores,
  component rows, and variance ratios normalized by **all** activity variance.
  Pass currents or rates explicitly according to the figure being reproduced.

Numerical checks compare factored/dense products, an analytic driven leak,
independent SciPy integration, Gaussian moments, and PCA reconstruction.
