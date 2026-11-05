# Gaussian mean-field theory

`lowrank.meanfield` solves the tanh model in Mastrogiuseppe & Ostojic (2018).
All covariances are **centered**; input strength is included in the input moments.
All variances describe activations `x`, not rates `tanh(x)`.

- `GaussianRankOne`: pattern means, standard deviations, centered `cov_mn`,
  random strength `g`, and input moments/covariances.
- `GaussianGeometry`: the same description for rank r; `cov_nm[j,l] = Cov(n_j,m_l)`.
  The n variances do not enter the DMF equations. Supply physically realizable
  joint Gaussian statistics (rank-one inputs are checked completely).
- `solve_fixed_points`: multistart solution of Eqs. 40/98, retaining unstable
  branches. Provide seeds to continue branches or explore rank r attractors.
- `solve_chaotic`: full Eq. 77 energy and long-lag conditions, including inputs;
  returns one branch, with temporal variance `variance - long_lag_variance`.
- `phase_boundaries`: Figure 1B for unit pattern standard deviations and
  `mean_m = mean_n = sqrt(structure_strength)`; Eq. 88 uses the chaotic state.

```python
from lowrank.meanfield import GaussianRankOne, solve_fixed_points

states = solve_fixed_points(GaussianRankOne(g=.8, mean_m=1.5, mean_n=1.5))
positive = states[-1]
print(positive.mean, positive.variance, positive.stable)
```

Gaussian expectations use normalized Gauss–Hermite quadrature (200 points for
static solutions; 100 for chaos by default). Fixed-point bulk stability uses
Eq. 44; collective stability uses Eqs. 71/99, extended to joint Gaussian rank r.
`outlier` is the largest real part of the collective mean-field eigenvalues;
it predicts an isolated network outlier when one exists. Extra mean-field
modes need not correspond to isolated network eigenvalues. A fixed point is
stable only if both this value and the bulk radius are below one.

Multistart solving does not guarantee finding every branch. Continuous ring
attractors cannot be enumerated. The chaotic solver rejects amplitudes below
1e-4, becomes poorly conditioned near onset, and does not establish chaotic
branch stability. Increase quadrature order to check high-variance cases.
These are infinite-size Gaussian predictions; finite networks fluctuate.
