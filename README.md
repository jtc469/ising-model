# Ising Model

## How do we estimate pi?

We generate N random points uniformly in the unit square, ($0 \leq x,y < 1$). A point lies inside a quarter-circle of radius r if ($x^2+y^2<r^2$), where ($0<r\leq1$).

The proportion of points inside the quarter-circle approximates its area divided by the square’s area:
$$
\frac{\text{points inside}}{N}\approx\frac{\pi r^2}{4}.
$$


Rearranging gives our estimate:
$$
\hat{\pi}=\frac{4}{r^2}\frac{\text{points inside}}{N}.
$$

```python
def estimate_pi(N, r=1.0):
    x = rng.random(N)
    y = rng.random(N)
    inside = np.count_nonzero(x**2 + y**2 < r**2)
    return 4 / r**2 * inside / N
```

## How accurate are the Monte Carlo estimates?

*These results use 2,000 Monte Carlo estimates for each choice of N and r. Re-running the notebook may give slightly different results*

The number of points inside the quarter-circle is binomial, so for large N the distribution is approximately Gaussian so I chose to fit it to that.

Across $\text{repeats}=2,000$ Monte-Carlo estimates with $N=10,000$, the mean pi estimate was $\hat\pi = 3.1413$, with standard deviation $\hat\sigma = 0.0166$. The expected standard deviation was $\sigma = 0.0164$ so results closely match the theory.

A standard deviation measures the uncertainty of one Monte Carlo simulation with N points. It scales with $\frac{1}{\sqrt N}$ i.e the uncertainty of the average of 2,000 independent estimates is smaller by $\sqrt 2000$



![pi histogram](out/plots/pi_histogram.png)


From $\sigma = 4\sqrt{\frac{p(1-p)}{N}} = CN^{-1/2}$, where $C=4\sqrt{p(1-p)}$ and $p=\frac{\pi r^2}{4}$, taking logs gives $$\log \sigma = -\frac{1}{2}\log N + \log C$$

Hence we expect the relationship $\sigma \propto N^{-1/2}$. This means fitting $\log \sigma$ vs $\log N$ should give a gradient of $-0.5$. We find $\text{measured slope} = -0.502$ so our simulation scales with theory.

![pi scaling](out/plots/pi_scaling.png)

Comparing two circle radii, we find 

| Radius squared | Relative SD $\times \sqrt{N}$ | Expected value |
| ---: | ---: | ---: |
| 1.000 | 0.512 | 0.523 |
| 0.667 | 0.957 | 0.954 |


where $\text{relative SD}= \frac{SD(\hat\pi)}{\pi}$. As the spread decreases with $\frac{1}{\sqrt{N}}$, we multiply by $\sqrt{N}$ to remove that dependence. The predicted value is

$$\text{relative error} \times \sqrt{N} = \sqrt{\frac{1-p}{p}}$$

Both measured values are close to their predictions. The smaller circle produces a larger relative spread, because fewer points land inside it.

For $N=10{,}000$, $\sqrt N=100$, so the measured relative spreads are 0.512% and 0.957%. The smaller circle therefore gives about 1.9 times as much spread for the same number of points.


## Simulating the Ising Model

### What is the Ising Model? 

It's a simple model of magnetism. In one dimension, it's like a row of tiny magnets. Each one can point in only two directions. Each spin only interacts with it's immediate neighbours.

`↑  ↑  ↓  ↑  ↓  ↓  ↑`

Neighbouring spins have lower energy when they point the same way, i.e an entire aligned row have the lowest possible energy. $J>0$ measures how strongly neighbours interact. Temperature affects how much higher-energy arrangements occur:
- At low temperature, the system strongly favours low-energy arrangements, so neighbours tend to align.
- At high temperature, higher-energy arrangements become more common, so the spins are more disordered.

We join the ends of the rows together to form a ring.

### What is the Metropolis Algorithm?

We need a way to generate spin arrangements with the right probabilities for a given temperature. The Metropolis algorithm does this by repeatedly proposing a small change and deciding whether to accept it.

Each step is:
1. Pick a spin at random.
2. Calculate how much the total energy would change if it flipped. Call this change $\Delta E=E_{\text{after}}-E_{\text{before}}$.
3. If the energy decreases or stays the same, accept the flip.
4. If the energy increases, accept it with probability$P=e^{-\Delta E/T}$

To simplify, we just use units where Boltzmann’s constant $k_B=1$

### Why does this give the right probabilities?

At a fixed temperature, a state with energy $E$ should show up with probability proportional to $e^{-E/T}$. We can't work these probabilities out directly because a ring of $N$ spins has $2^N$ states, which is far too many to add up.

The Metropolis rule gets around this. Take any two states A and B, where B has the higher energy. The move from A to B is accepted with probability $e^{-\Delta E/T}$ and the move back is always accepted. So in the long run

$$\frac{P(B)}{P(A)}=e^{-\Delta E/T}$$

which is exactly the ratio we want. This is called detailed balance. We only ever need the ratio of two probabilities, so the sum over all $2^N$ states never has to be calculated.

### How do we simulate the 1D model?

The energy of the ring is

$$E=-J\sum_{i} s_i s_{i+1}$$

where each bond is counted once and I set $J=1$. The project sheet writes this with a factor of $\frac{1}{2}$ because its sum counts every pair twice. I put the minus sign in so that aligned spins have the lower energy when $J$ is positive.

Flipping one spin only changes the two bonds next to it, so

$$\Delta E=2Js_i(s_{i-1}+s_{i+1})$$

This means we never have to recalculate the whole energy just to decide on one flip.

```python
def sweep_1d(spins, T):
    N = len(spins)
    sites = rng.integers(N, size=N)
    randoms = rng.random(N)

    for i, u in zip(sites, randoms):
        dE = 2 * spins[i] * (spins[i - 1] + spins[(i + 1) % N])

        if dE <= 0 or u < np.exp(-dE / T):
            spins[i] = -spins[i]
```

One sweep is $N$ attempted flips, so on average every spin gets one chance to flip. At each temperature I
1. Start with all spins up.
2. Run 1,000 sweeps and ignore them, so the system forgets where it started.
3. Run 5,000 more sweeps and measure the energy after each one.

I start with the spins aligned because at very low temperatures a random start gets stuck with domain walls that take a very long time to disappear.

### What do we measure?

The internal energy per spin is the average energy divided by the number of spins, $e(T)=\langle E\rangle/N$.

The specific heat tells us how much the energy changes when the temperature changes. It can also be found from how much the energy fluctuates at a fixed temperature:

$$C(T)=\frac{\langle E^2\rangle-\langle E\rangle^2}{Nk_BT^2}$$

This is useful because both quantities come from the same list of measured energies. The top line is just the variance of the energies, so in the code it's `np.var(energies)`.

### How does it compare to the exact result?

The 1D model can be solved exactly with a transfer matrix. The partition function $Z$ is a sum over all states of a product of terms $e^{Ks_is_{i+1}}$, one for each bond, where $K=J/T$. Each term depends on two neighbouring spins which have two values each, so it can be written as a 2x2 matrix:

```math
\mathbf{T}=\begin{pmatrix} e^{K} & e^{-K} \\ e^{-K} & e^{K}\end{pmatrix}
```

Summing over a spin is the same as multiplying two of these matrices together, and going all the way round the ring gives $Z=\text{Tr}(\mathbf{T}^N)=\lambda_+^N+\lambda_-^N$. The eigenvalues are $\lambda_+=2\cosh K$ and $\lambda_-=2\sinh K$. For large $N$ only the bigger one matters so $\ln Z\approx N\ln(2\cosh K)$, which gives

$$e(T)=-J\tanh(J/T)$$

$$C(T)=\frac{(J/T)^2}{\cosh^2(J/T)}$$

![1d energy](out/plots/ising_1d_energy.png)

![1d specific heat](out/plots/ising_1d_specific_heat.png)

The energy sits right on the exact curve at every temperature. The specific heat is noisier because it's a variance rather than a mean, but it follows the exact curve too.

Two things to notice:
- The bump in the specific heat near $T\approx0.8J$ is smooth. It isn't a phase transition, the 1D model doesn't have one. A single domain wall only costs $2J$ of energy but it can sit in $N$ different places, so at any temperature above zero the walls appear and break up the order.
- Around $T=0.3J$ the simulation gives $C=0$ but the exact value is about 0.04. Here the chance of accepting a flip out of the aligned state is about $e^{-4/0.3}\approx10^{-6}$, so in 6,000 sweeps of 100 spins it basically never happens. The algorithm isn't wrong, it's just frozen.

## The 2D Ising Model

### What changes in 2D?

The spins now sit on an $L\times L$ square grid and each one has four neighbours (up, down, left and right). The edges wrap round in both directions. The algorithm is exactly the same as before, the only difference is that $\Delta E$ uses four neighbours instead of two.

```python
neighbours = (spins[i - 1, j] + spins[(i + 1) % L, j]
              + spins[i, j - 1] + spins[i, (j + 1) % L])
dE = 2 * spins[i, j] * neighbours
```

I use a 20x20 lattice at 40 temperatures, with 1,000 sweeps to settle and 3,000 sweeps of measurements.

As well as the specific heat we measure the magnetisation per spin, $m=\frac{1}{N}\sum s_i$. It's 1 if every spin points the same way and about 0 if they are random.

I average $|m|$ rather than $m$. Up and down are equally likely, so on a small lattice the whole thing sometimes flips from mostly up to mostly down. The average of $m$ would then come out near zero even when the spins are clearly ordered.

### Is there a phase transition?

Yes, unlike in 1D.

![2d magnetisation](out/plots/ising_2d_magnetisation.png)

![2d specific heat](out/plots/ising_2d_specific_heat.png)

Below about $T=1.5J$ nearly all the spins are aligned. Then the magnetisation drops quickly and the specific heat has a sharp peak. Onsager solved the 2D model exactly for an infinite lattice and found

$$T_c=\frac{2J}{\ln(1+\sqrt2)}\approx2.269J$$

In my simulation the specific heat peaks at $T=2.36J$, and the magnetisation follows Onsager's curve below $T_c$.

It doesn't match exactly because my lattice is finite:
- The peak in $C$ is rounded and sits a bit above $T_c$. On an infinite lattice it would go to infinity.
- $|m|$ doesn't go all the way to zero above $T_c$. $N$ random spins add up to roughly $\sqrt N$, so there's a tail of about $1/\sqrt N=0.05$.

![2d snapshots](out/plots/ising_2d_snapshots.png)

These are snapshots of a 40x40 lattice. Below $T_c$ almost everything points the same way, above $T_c$ there are only small patches, and near $T_c$ there are patches of every size.

So why does 2D have a transition when 1D doesn't? In 2D a domain wall is a line, so the energy it costs grows with the size of the region that flips. At low temperatures that cost wins and the order survives.

### What is the correlation length?

The patches in the snapshots have a typical size, which is called the correlation length $\xi$. To measure it we ask: if one spin is up, how likely is a spin $r$ sites away to be up as well? Above $T_c$ this falls off as

$$G(r)\sim e^{-r/\xi}$$

As the temperature gets closer to $T_c$ the patches get bigger. Theory says $\xi$ grows as a power law

$$\xi\propto(T-T_c)^{-\nu}$$

with $\nu=1$ for the 2D Ising model.

To measure $G(r)$ I average the spins in each row, multiply each row average by the one $r$ rows away, and then average over the lattice and over 5,000 sweeps. I do the same with the columns to get twice as much data.

```python
row_means = np.mean(spins, axis=1)
col_means = np.mean(spins, axis=0)

for r in range(r_max + 1):
    G[r] += np.mean(row_means * np.roll(row_means, r))
    G[r] += np.mean(col_means * np.roll(col_means, r))
```

I use rows instead of single spins because the single spin version has an extra factor of roughly $1/\sqrt r$ in front of the exponential. That makes a straight line fit give a $\xi$ that is too small (I tried it and got $\nu\approx0.65$). Averaging over the row removes that factor.

*These results use a 40x40 lattice at 8 temperatures above $T_c$.*

![2d correlation](out/plots/ising_2d_correlation.png)

On a log scale $G(r)$ is a straight line with gradient $-1/\xi$. The lines get shallower as we get closer to $T_c$, i.e the correlation length is growing. At large $r$ the points get noisy, so I only fit the points where $G(r)$ is more than 20% of $G(0)$.

| $T/J$ | Measured $\xi$ | Exact $\xi$ |
| ---: | ---: | ---: |
| 2.5 | 6.34 | 5.96 |
| 2.6 | 4.39 | 4.27 |
| 2.7 | 3.43 | 3.37 |
| 2.8 | 2.78 | 2.80 |
| 2.9 | 2.43 | 2.42 |
| 3.0 | 2.09 | 2.14 |
| 3.2 | 1.73 | 1.75 |
| 3.4 | 1.52 | 1.51 |

The exact values are for an infinite lattice, where $1/\xi=\ln\coth(J/T)-2J/T$.

We then use the same trick as for pi. Taking logs of the power law gives

$$\log\xi=-\nu\log(T-T_c)+\text{const}$$

so fitting $\log\xi$ vs $\log(T-T_c)$ should give a gradient of $-\nu$. We find $\nu=0.91$.

![2d power law](out/plots/ising_2d_xi_power_law.png)

This is a bit below 1. The power law is only exact very close to $T_c$ and my temperatures go up to $3.4J$. Fitting the exact values of $\xi$ over the same temperatures gives 0.87, so the simulation is doing what it should. I can't go much closer to $T_c$ because $\xi$ would get close to the size of the lattice, and the simulation also takes much longer to settle there.
