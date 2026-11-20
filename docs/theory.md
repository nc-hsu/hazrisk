# Theory

This page sets out the quantities hazrisk computes and the approximations
behind them, with the papers they come from (listed in full on the
{doc}`references` page). The function that implements each result is named
next to it; its docstring gives the same formula.

Throughout, $\ln$ is the natural logarithm, $im$ is the intensity measure (e.g.
a spectral acceleration in g), and a fragility is lognormal with median
capacity $\theta_c$ and dispersion $\beta$.

## The hazard curve

The hazard curve $H(im)$, written `H_im` in the code, is the mean annual
frequency of ground motions that exceed $im$ at the site. It decreases with
$im$. A PSHA program such as OpenQuake reports it at a few tens of levels, as
a frequency or as a probability of exceedance in an investigation time $t$.

If earthquakes, and so exceedances of $im$, occur as a Poisson process, an
event with mean annual frequency $\lambda$ (a level of the hazard curve,
$\lambda = H(im)$, or the frequency $\lambda_C$ of a limit state) happens at
least once in $t$ years with probability (Baker, Bradley & Stafford 2021,
Section 6.2.2, pp. 253–255)

$$
P(t) = 1 - e^{-\lambda t},
\qquad\text{so}\qquad
\lambda = -\frac{\ln(1 - P)}{t}.
$$

The mean return period is $T = 1/\lambda$. The module {mod}`hazrisk.convert`
converts between the three: {func}`~hazrisk.convert.poe_to_mafe`,
{func}`~hazrisk.convert.mafe_to_poe`, {func}`~hazrisk.convert.poe_to_rtp`,
{func}`~hazrisk.convert.rtp_to_poe`, {func}`~hazrisk.convert.mafe_to_rtp` and
{func}`~hazrisk.convert.rtp_to_mafe`. For example, 10 % in 50 years is
$\lambda = 2.107 \times 10^{-3}$ per year, a return period of 475 years. For
small $\lambda t$, $P \approx \lambda t$.

## First- and second-order approximations

The closed-form risk solutions need the hazard curve as a formula. The
**first-order** approximation is a straight line in log-log space, a power law
(Cornell et al. 2002; {func}`~hazrisk.hazard.first_order_approx`):

$$
H(im) = k_0 \, im^{-k_1}.
$$

A real hazard curve bends downwards in log-log space: its slope steepens as
$im$ grows. So a power law fits it only over a narrow range. The
**second-order** approximation of Vamvatsikos (2013) adds a quadratic term in
$\ln im$ ({func}`~hazrisk.hazard.second_order_approx`):

$$
H(im) = k_0 \exp\left(-k_2 (\ln im)^2 - k_1 \ln im\right),
$$

which is a parabola in log-log space and reduces to the first order at
$k_2 = 0$. Both have analytical derivatives
({func}`~hazrisk.hazard.first_order_dHdx`,
{func}`~hazrisk.hazard.second_order_dHdx`).

### Fitting

hazrisk fits the coefficients in two ways.

- **Through chosen points** ({func}`~hazrisk.hazard.fit_first_order_points`,
  {func}`~hazrisk.hazard.fit_second_order_points`): the curve passes exactly
  through two (first order) or three (second order) points of the numerical
  curve, given as intensities or as frequencies (e.g. the 475 and 2475 year
  levels). Between the tabulated levels the numerical curve is interpolated
  linearly in log-log space. The result depends on where the points are put.
- **By regression over a band** ({func}`~hazrisk.hazard.fit_first_order_regression`,
  {func}`~hazrisk.hazard.fit_second_order_regression`): a least-squares fit of
  $\ln H$ against $\ln im$, as a straight line or a parabola, over all
  the tabulated levels inside a band of intensities or of frequencies. This
  uses only the tabulated levels, so the band must hold at least two (first
  order) or three (second order) of them.

The regression functions also return the coefficient of determination of the
fit in $\ln H$,

$$
r^2 = 1 - \frac{\sum_i (\ln H_i - \ln \hat H_i)^2}
               {\sum_i (\ln H_i - \overline{\ln H})^2},
$$

where $H_i$ is the tabulated hazard at level $i$, $\hat H_i$ the fitted value
there, and the sums run over
the levels in the band. Because it is measured on the logarithm, a given $r^2$
means the same relative error at the frequent and at the rare end of the curve.

### The turnover of the second-order fit

The slope of the second-order curve in log-log space is

$$
\frac{d \ln H}{d \ln im} = -k_1 - 2 k_2 \ln im,
$$

which is zero at $\ln im = -k_1 / (2 k_2)$. For $k_2 > 0$ the fitted curve
rises to a peak there and, below that intensity, the hazard *increases* with
$im$, which is not physical. The turnover normally lies below the fitted band,
often far below, but on steep curves (large $k_2$) it can fall inside the
range of a plot. The fit functions do not check for it: the approximation
should be trusted only over the band it was fitted to.

## PCHIP interpolation of a tabulated curve

To integrate a tabulated hazard curve accurately, hazrisk refines it between
the tabulated levels with a piecewise cubic Hermite interpolating polynomial
(PCHIP; Fritsch & Butland 1984; Moler 2004) of $\ln H$ against $\ln im$,
using scipy's {class}`~scipy.interpolate.PchipInterpolator`
({func}`~hazrisk.hazard.pchip_interpolant`,
{func}`~hazrisk.hazard.pchip_approx`, {func}`~hazrisk.hazard.pchip_dHdx`).
PCHIP passes through every tabulated point, has a continuous first derivative,
and is monotone wherever the data are, so the refined hazard still decreases
and does not overshoot between levels.

The risk integrand (below) contains the slope of the hazard curve. With
straight lines between the tabulated levels in log-log space, that slope is
constant over each interval and jumps at every level, so the integrand has
steps and the integral is biased where the slope changes fastest: at the rare
end of the curve. PCHIP keeps the slope
continuous. For a power-law hazard curve the log-log data lie on a straight
line, PCHIP reproduces it exactly, and the numerical result converges to the
first-order closed form.

## The risk integral

The mean annual frequency of the limit state is the total probability theorem
applied over all intensities: the frequency of ground motions at each
intensity times the probability of the limit state at that intensity,

$$
\lambda_C = \int_0^\infty P(C \mid im) \left| \frac{dH}{d\,im} \right| d\,im
$$

(Baker, Bradley & Stafford 2021, Section 9.4.1, pp. 382–385).

### The change of variable in `numerical_mafe`

{func}`~hazrisk.risk.numerical_mafe` does not estimate the derivative
$dH/d\,im$. Because $H$ decreases monotonically with $im$,
$\left|dH/d\,im\right| d\,im = -dH$, and the integral becomes

$$
\lambda_C = -\int P(C \mid im) \, dH,
$$

the area under the fragility plotted against the hazard. This is evaluated
with the trapezoid rule over the tabulated pairs $(H_i, P_i)$, ordered by
increasing $im$. The trapezoid rule then acts on the smooth, bounded fragility
rather than on the sharply peaked hazard derivative, and $im$ itself drops out,
so uneven spacing of the levels is handled automatically.

The tabulated curve stops at its last level $im_{n-1}$. The ground motions
above it have frequency $H_{n-1}$, and the **tail term** counts them
with the fragility held at its last value:

$$
\lambda_C \approx \sum_{i=0}^{n-2} \tfrac{1}{2} (P_i + P_{i+1})
                    (H_i - H_{i+1})
                  + H_{n-1} P_{n-1}.
$$

If the curve ends in zeros, as OpenQuake writes it, the tail term is zero and
the last trapezoid runs from the last positive level down to the first zero
level (see {doc}`examples/trailing_zeros`). Below the first level $im_0$ the
fragility is taken as negligible.

{func}`~hazrisk.risk.numerical_mafe_pchip` applies this to a lognormal
fragility on a PCHIP-refined curve (2000 log-spaced levels by default, up to
the last positive level, followed by the first zero level if there is one),
which removes the bias of the coarse PSHA grid.

## Closed-form solutions (SAC/FEMA)

With a lognormal fragility and a fitted hazard curve, the risk integral has a
closed form. These are the solutions behind the SAC/FEMA framework.

### First order

On the first-order hazard, the risk is (Cornell et al. 2002)

$$
\lambda_C = k_0 \, \theta_c^{-k_1} \exp\left(\tfrac{1}{2} k_1^2 \beta^2\right)
          = H(\theta_c) \exp\left(\tfrac{1}{2} k_1^2 \beta^2\right),
$$

({func}`~hazrisk.risk.mafe_first_order`): the hazard at the median capacity,
increased by a factor that grows with the dispersion and with the slope of the
hazard curve. Solving for the capacity gives the median capacity required for a
target frequency $\lambda_C$ ({func}`~hazrisk.risk.required_capacity_first_order`):

$$
\theta_c = \left(\frac{k_0}{\lambda_C}\right)^{1/k_1}
           \exp\left(\tfrac{1}{2} k_1 \beta^2\right).
$$

### Second order

On the second-order hazard, Vamvatsikos (2013) gives

$$
\lambda_C = \sqrt{p}\, k_0^{1-p}\, H(\theta_c)^p
            \exp\left(\tfrac{1}{2} p k_1^2 \beta^2\right),
\qquad
p = \frac{1}{1 + 2 k_2 \beta^2}
$$

({func}`~hazrisk.risk.mafe_second_order`, {func}`~hazrisk.risk.p_second_order`),
with $H(\theta_c)$ the second-order curve at the median capacity. At
$k_2 = 0$, $p = 1$ and this is the first-order result.

(theory-required-capacity-second-order)=
### Required capacity, second order

The median capacity that gives a target $\lambda_C$ on the second-order hazard
({func}`~hazrisk.risk.required_capacity_second_order`) follows from the closed
form above (Vamvatsikos 2013). Write $x = \ln\theta_c$, so that

$$
\ln H(\theta_c) = \ln k_0 - k_1 x - k_2 x^2 .
$$

Taking logs of the closed form and writing

$$
L = \ln\frac{\lambda_C}{k_0 \sqrt{p}}
$$

gives a quadratic in $x$,

$$
k_2 x^2 + k_1 x + \left(\frac{L}{p} - \tfrac{1}{2} k_1^2 \beta^2\right) = 0 .
$$

Its discriminant, simplified with $1/p = 1 + 2 k_2 \beta^2$, is

$$
D = \frac{k_1^2 - 4 k_2 L}{p} .
$$

A negative $D$ means no capacity gives the target: the fitted second-order
curve never reaches that frequency (see the range of validity below). Of the
two roots, the one that tends to the first-order answer as $k_2 \to 0$ is

$$
x = \frac{-k_1 + \sqrt{D}}{2 k_2} .
$$

Written like that it divides by $k_2$: it fails at $k_2 = 0$ and loses
accuracy for small $k_2$, where $\sqrt{D}$ is close to $k_1$ and the numerator
cancels. Multiplying the numerator and the denominator by $k_1 + \sqrt{D}$, and
using $D - k_1^2 = 2 k_2 (k_1^2 \beta^2 - 2L/p)$, removes the division:

$$
x = \frac{k_1^2 \beta^2 - 2L/p}{k_1 + \sqrt{D}}
  = \frac{\tfrac{1}{2} k_1 \beta^2 - L / (p k_1)}
         {(k_1 + \sqrt{D}) / (2 k_1)} .
$$

The second form is the one evaluated. At $k_2 = 0$, $p = 1$, $\sqrt{D} = k_1$
and the denominator is exactly one, so the result is exactly the first-order
required capacity.

### Range of validity

- The closed forms are only as good as the fitted hazard over the intensities
  that dominate the risk, roughly from $\theta_c$ down to a few $\beta$ below
  it. Fit over a band that covers them; {doc}`examples/risk_contribution`
  shows how to find that band.
- The second-order solution needs $1 + 2 k_2 \beta^2 > 0$. This always holds
  for $k_2 \ge 0$, the usual case; hazrisk raises a `ValueError` otherwise.
- Because the fitted second-order curve has a peak at its turnover, the
  closed-form frequency of the limit state has a maximum too, reached when
  $\theta_c$ is at the turnover. A target above that maximum has no solution,
  and {func}`~hazrisk.risk.required_capacity_second_order` raises a
  `ValueError`. Below the maximum the quadratic has a second root, on the
  non-physical side of the turnover; hazrisk returns the root that tends to
  the first-order answer as $k_2 \to 0$.
- Both assume a lognormal fragility.

## Risk ratios and capacity ratios

Compare two designs on the same first-order hazard, with medians
$\theta_1, \theta_2$ and dispersions $\beta_1, \beta_2$. The first-order closed
form gives the risk of each,

$$
\lambda_1 = k_0 \, \theta_1^{-k_1} \exp\left(\tfrac{1}{2} k_1^2 \beta_1^2\right),
\qquad
\lambda_2 = k_0 \, \theta_2^{-k_1} \exp\left(\tfrac{1}{2} k_1^2 \beta_2^2\right).
$$

Dividing the two, $k_0$ cancels, and with $R_\theta = \theta_2/\theta_1$ and
$R_\beta = \beta_2/\beta_1$ (so $\beta_2^2 = R_\beta^2 \beta_1^2$) the ratio of
their risks is ({func}`~hazrisk.risk.risk_ratio_first_order`)

$$
R_\lambda = \frac{\lambda_2}{\lambda_1}
          = R_\theta^{-k_1}
            \exp\left(\tfrac{1}{2} k_1^2 \beta_1^2 (R_\beta^2 - 1)\right).
$$

Solved for $R_\theta$, this is the change of median capacity needed for a
target change of risk ({func}`~hazrisk.risk.theta_ratio_first_order`):

$$
R_\theta = \exp\left(\tfrac{1}{2} k_1 \beta_1^2 (R_\beta^2 - 1)
                     - \frac{\ln R_\lambda}{k_1}\right).
$$

For example, with equal dispersions ($R_\beta = 1$) the first term in the
exponent is zero, and halving the risk means $R_\lambda = 1/2$, so
$\ln R_\lambda = -\ln 2$ and

$$
R_\theta = \exp\left(\frac{\ln 2}{k_1}\right) = 2^{1/k_1},
$$

which is about 1.26, or 26 % more capacity, for $k_1 = 3$.

## Risk contribution

Which intensities does the risk come from? Written per unit of $\ln im$, the
integrand of the risk integral is
({func}`~hazrisk.risk.risk_contribution_pchip`)

$$
\frac{d\lambda_C}{d \ln im} = P(C \mid im) \left|\frac{dH}{d\,im}\right| im,
$$

with the slope of the hazard from the PCHIP interpolant. Plotted against $im$
on a log axis, the area under this curve is the risk. Per unit $\ln im$
rather than per unit $im$ because hazard curves are tabulated and plotted on
log-spaced levels.

The discrete counterpart, {func}`~hazrisk.risk.risk_contribution`, splits the
trapezoid sum of `numerical_mafe` into its terms: one per interval between
tabulated levels,

$$
\Delta\lambda_{C,i} = \tfrac{1}{2} (P_i + P_{i+1}) (H_i - H_{i+1}),
$$

the frequency of ground motions in the interval times the average probability
of the limit state over it, plus the tail term $H_{n-1} P_{n-1}$ as the
last element. They add up to the result of `numerical_mafe`. Either function can
return the contributions as frequencies or normalised by the total.
