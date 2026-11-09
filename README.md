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
    x = seed.random(N)
    y = seed.random(N)
    inside = np.count_nonzero(x**2 + y**2 < r**2)
    return 4 / r**2 * inside / N
```

## How accurate is the Monte Carlo estimates?

*This report uses data from  2,000 runs Monte-Carlo estimates, each using N = 10,000 points per estimate. Re-running the notebook may give slightly different results*

![pi histogram](out/plots/pi_histogram.png)

Across $\text{repeats}=2,000$ Monte-Carlo estimates with $N=10,000$, the mean pi estimate was $\hat\pi = 3.1407$, with standard deviation $\hat\sigma = 0.0165$. The calculated expected standard deviation was $\sigma = 0.0164$ (percentage error of just 0.6%) so results closely match the theory.

From $\sigma = 4\sqrt{\frac{p(1-p)}{N}} = CN^{-1/2}$, where $C=4\sqrt{p(1-p)}$, taking logs gives $$\log \sigma = -\frac{1}{2}\log N + \log C$$

Hence we expect the relationship $\sigma \propto N^{-1/2}$. This means fitting sample_sizes vs sigmas should give a gradient of $-0.5$. We find $\text{measured slope} = -0.510$ so our simulation scales with theory.

Comparing radii/box sizes, we find 

| Radius squared | Relative error $\times \sqrt{N}$ | Expected value |
| ---: | ---: | ---: |
| 1.000 | 0.539 | 0.523 |
| 0.667 | 0.967 | 0.954 |


where $\text{relative error}= \frac{SD(\hat\pi)}{\pi}$. As the spread decreases with $\frac{1}{N}$, we multiply by $\sqrt{N}$ to remove that dependence. The predicted value is

$$\text{relative error} \times \sqrt{N} = \sqrt{\frac{1-p}{p}}$$

$$p=\frac{\pi r^2}{4}$$.

Both measured values are close to their predictions. The smaller circle produces a larger relative spread, because fewer points land inside it.

For $N=10{,}000$, $\sqrt N=100$, so the measured relative spreads are 0.539% and 0.967%. The smaller circle therefore gives about 1.8 times as much spread for the same number of points.
