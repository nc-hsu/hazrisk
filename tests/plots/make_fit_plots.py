"""Worked examples of the hazard-curve fitting functions, on real hazard data.

Run from the repository root:

    .venv\\Scripts\\python.exe tests/plots/make_fit_plots.py

Writes the figures next to this script. The data is a set of AvgSA hazard curves
for 60 sites; the second column of each curve is taken as the mean annual
frequency of exceeding the intensity measure level in the first.

These are illustrations, not tests -- pytest ignores this file (its name does not
start with ``test_``).
"""
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np

from hazrisk import hazard as haz

DATA = Path(r"C:\Users\clemettn\Documents\phd\data_processed\03_site_hazard"
            r"\AvgSA_03_hazard_curves_60sites_4sig.pickle")
HERE = Path(__file__).parent

# The three hazard levels an engineer would usually anchor a fit on.
RTP_LABELS = {475: "10%/50yr", 975: "5%/50yr", 2475: "2%/50yr"}

# Return-period range the regression figures fit over. Figure names carry this
# label, so re-running with a different band leaves the previous set in place.
BAND_RTP = (10, 10000)
BAND = (1 / BAND_RTP[1], 1 / BAND_RTP[0])
BAND_LABEL = f"{BAND_RTP[0]}-{BAND_RTP[1]}yr"


def load_curve(site: int, stat: str = "mean") -> tuple[np.ndarray, np.ndarray]:
    """Return the (im, H_im) hazard curve for one site, trimmed for fitting.

    The raw curves run out to intensities where the hazard has underflowed to
    zero (and, at the very small values, is no longer strictly decreasing). The
    fitting functions require a strictly positive, strictly decreasing curve, so
    the flat tail is trimmed here -- which is the trimming any caller has to do
    with real data.
    """
    with open(DATA, "rb") as f:
        curves = pickle.load(f)

    im, H_im = curves[site]["AvgSA"][stat].T

    keep = H_im > 0
    im, H_im = im[keep], H_im[keep]

    decreasing = np.concatenate([[True], np.diff(H_im) < 0])
    stop = np.argmin(decreasing) if not decreasing.all() else len(im)
    return im[:stop], H_im[:stop]


def _hazard_axes(ax, im, H_im, title):
    ax.loglog(im, H_im, "o-", color="0.25", ms=3, lw=1.2, zorder=5,
              label="numerical hazard curve")
    ax.set_xlabel("AvgSA [g]")
    ax.set_ylabel("mean annual frequency of exceedance")
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", lw=0.4, alpha=0.4)


def _mark(ax, im_pts, mafe_pts, color, label=None):
    ax.plot(im_pts, mafe_pts, "*", color=color, ms=14, mec="k", mew=0.5,
            zorder=10, label=label)


def figure_points_vs_regression(im, H_im, site):
    """Exact fits through chosen points, next to least-squares fits over a band."""
    mafe_anchors = np.array([1 / 475, 1 / 975, 1 / 2475])
    # A point fit interpolates, so it can be anchored anywhere on the curve. A
    # regression can only use the levels the curve actually provides, and this
    # curve has just 25 of them -- the 475-2475 yr band holds only two, too few
    # for either fit, hence the wider BAND.
    band = BAND
    fine = np.logspace(np.log10(im[0]), np.log10(im[-1]), 300)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)

    # -- left: exact fits through the anchor points
    ax = axes[0]
    _hazard_axes(ax, im, H_im, f"site {site}: fits through chosen MAFE points")

    k0, k1 = haz.fit_first_order_points(im, H_im, mafe=mafe_anchors[[0, 2]])
    ax.loglog(fine, haz.first_order_approx(fine, k0, k1), "--", color="C0",
              label=f"1st order, 2 points\nk0={k0:.3g}, k1={k1:.3g}")

    k0, k1, k2 = haz.fit_second_order_points(im, H_im, mafe=mafe_anchors)
    ax.loglog(fine, haz.second_order_approx(fine, k0, k1, k2), "-", color="C3",
              label=f"2nd order, 3 points\nk0={k0:.3g}, k1={k1:.3g}, k2={k2:.3g}")

    iml_anchors = np.interp(np.log(mafe_anchors), np.log(H_im)[::-1],
                            np.log(im)[::-1])
    _mark(ax, np.exp(iml_anchors), mafe_anchors, "gold",
          "anchors: 10%, 5%, 2% in 50yr")
    ax.legend(fontsize=7, loc="lower left")

    # -- right: least-squares fits over the same MAFE band
    ax = axes[1]
    _hazard_axes(ax, im, H_im,
                 f"site {site}: regression over the {BAND_LABEL} band")

    k0, k1, r2 = haz.fit_first_order_regression(im, H_im, mafe_band=band)
    ax.loglog(fine, haz.first_order_approx(fine, k0, k1), "--", color="C0",
              label=f"1st order regression\nk0={k0:.3g}, k1={k1:.3g}, r2={r2:.4f}")

    k0, k1, k2, r2 = haz.fit_second_order_regression(im, H_im, mafe_band=band)
    ax.loglog(fine, haz.second_order_approx(fine, k0, k1, k2), "-", color="C3",
              label=f"2nd order regression\nk0={k0:.3g}, k1={k1:.3g}, "
                    f"k2={k2:.3g}, r2={r2:.4f}")

    ax.axhspan(band[0], band[1], color="gold", alpha=0.15, label="fitting band")
    ax.legend(fontsize=7, loc="lower left")

    ax.set_ylim(H_im[-1] / 3, H_im[0] * 3)
    fig.tight_layout()
    fig.savefig(HERE / f"01_points_vs_regression_{BAND_LABEL}.png", dpi=150)
    plt.close(fig)


def figure_anchor_sensitivity(im, H_im, site):
    """How much the first order k-values depend on where the fit is anchored."""
    pairs = [(475, 2475), (75, 475), (2475, 10000)]
    fine = np.logspace(np.log10(im[0]), np.log10(im[-1]), 300)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    _hazard_axes(ax, im, H_im,
                 f"site {site}: first order fit is only as good as its anchors")

    for colour, (lo_rtp, hi_rtp) in zip(("C0", "C1", "C2"), pairs):
        mafe = np.array([1 / lo_rtp, 1 / hi_rtp])
        k0, k1 = haz.fit_first_order_points(im, H_im, mafe=mafe)
        ax.loglog(fine, haz.first_order_approx(fine, k0, k1), "--", color=colour,
                  label=f"{lo_rtp}-{hi_rtp} yr: k0={k0:.3g}, k1={k1:.3g}")
        iml = np.exp(np.interp(np.log(mafe), np.log(H_im)[::-1], np.log(im)[::-1]))
        _mark(ax, iml, mafe, colour)

    ax.set_ylim(H_im[-1] / 3, H_im[0] * 3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(HERE / "02_anchor_sensitivity.png", dpi=150)
    plt.close(fig)


def figure_band_width(im, H_im, site):
    """r2 of both orders as the regression band is widened."""
    bands = [(1 / 2475, 1 / 250), (1 / 5000, 1 / 100), (1 / 10000, 1 / 10),
             (H_im[-1], H_im[0])]
    labels = ["250-2475 yr", "100-5000 yr", "10-10000 yr", "whole curve"]
    fine = np.logspace(np.log10(im[0]), np.log10(im[-1]), 300)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, order in zip(axes, (1, 2)):
        _hazard_axes(ax, im, H_im, f"site {site}: order {order} regression band")
        for colour, band, label in zip(("C0", "C1", "C2", "C3"), bands, labels):
            if order == 1:
                k0, k1, r2 = haz.fit_first_order_regression(im, H_im,
                                                            mafe_band=band)
                fitted = haz.first_order_approx(fine, k0, k1)
                text = f"{label}: k1={k1:.3g}, r2={r2:.4f}"
            else:
                k0, k1, k2, r2 = haz.fit_second_order_regression(im, H_im,
                                                                 mafe_band=band)
                fitted = haz.second_order_approx(fine, k0, k1, k2)
                text = f"{label}: k1={k1:.3g}, k2={k2:.3g}, r2={r2:.4f}"
            ax.loglog(fine, fitted, "-", color=colour, lw=1.4, label=text)
        ax.set_ylim(H_im[-1] / 3, H_im[0] * 3)
        ax.legend(fontsize=7, loc="lower left")

    fig.tight_layout()
    fig.savefig(HERE / "03_band_width.png", dpi=150)
    plt.close(fig)


def figure_site_grid(sites):
    """Second order regression over the design band, across several sites."""
    band = BAND
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))

    for ax, site in zip(axes.ravel(), sites):
        im, H_im = load_curve(site)
        fine = np.logspace(np.log10(im[0]), np.log10(im[-1]), 300)
        _hazard_axes(ax, im, H_im, f"site {site}")

        k0, k1, r2 = haz.fit_first_order_regression(im, H_im, mafe_band=band)
        ax.loglog(fine, haz.first_order_approx(fine, k0, k1), "--", color="C0",
                  label=f"1st: k1={k1:.2f}, r2={r2:.4f}")

        k0, k1, k2, r2 = haz.fit_second_order_regression(im, H_im, mafe_band=band)
        ax.loglog(fine, haz.second_order_approx(fine, k0, k1, k2), "-", color="C3",
                  label=f"2nd: k1={k1:.2f}, k2={k2:.2f}, r2={r2:.4f}")

        ax.axhspan(band[0], band[1], color="gold", alpha=0.15)
        ax.set_ylim(H_im[-1] / 3, H_im[0] * 3)
        ax.legend(fontsize=7, loc="lower left")

    fig.suptitle(f"Regression over the {BAND_LABEL} band, six sites", fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / f"04_site_grid_{BAND_LABEL}.png", dpi=150)
    plt.close(fig)


def main():
    site = 0
    im, H_im = load_curve(site)
    figure_points_vs_regression(im, H_im, site)
    figure_anchor_sensitivity(im, H_im, site)
    figure_band_width(im, H_im, site)
    figure_site_grid([0, 12, 24, 36, 48, 59])
    print(f"figures written to {HERE}")


if __name__ == "__main__":
    main()
