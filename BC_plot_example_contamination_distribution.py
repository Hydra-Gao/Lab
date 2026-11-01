import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Parameters
# ============================================================

RECORDING_DURATION_S = 2667.625

# Spike counts: 100 -> 10000, linearly spaced
spike_counts = np.linspace(100, 10000, 1000)

# SI defaults / your current settings
RP_MS = 1.0
CENSORED_MS = 0.0

ISI_THRESHOLD_MS = 1.5
MIN_ISI_MS = 0.0


# Convert ms -> s
rp_s = RP_MS / 1000
censored_s = CENSORED_MS / 1000

isi_threshold_s = ISI_THRESHOLD_MS / 1000
min_isi_s = MIN_ISI_MS / 1000


# ============================================================
# Functions
# ============================================================

def compute_rp_contamination(
    n_violations,
    n_spikes,
    recording_duration_s,
    refractory_period_s,
    censored_period_s=0.0,
):
    """
    SI-style refractory-period contamination estimate.

    contamination =
        1 - sqrt(
            1 - n_violations * T
                / (N^2 * (RP - censored))
        )

    If the term inside sqrt becomes negative,
    the contamination estimate is not defined -> NaN.
    """

    x = (
        n_violations * recording_duration_s
        / (
            n_spikes**2
            * (refractory_period_s - censored_period_s)
        )
    )

    inside_sqrt = 1 - x

    contamination = np.full_like(n_spikes, np.nan, dtype=float)

    valid = inside_sqrt >= 0

    contamination[valid] = (
        1 - np.sqrt(inside_sqrt[valid])
    )

    return contamination


def compute_isi_violations_ratio(
    n_violations,
    n_spikes,
    recording_duration_s,
    isi_threshold_s,
    min_isi_s=0.0,
):
    """
    SI-style ISI violations ratio.

    isi_ratio =
        n_violations * T
        / (
            2 * N^2
            * (ISI_threshold - min_ISI)
        )
    """

    return (
        n_violations * recording_duration_s
        / (
            2
            * n_spikes**2
            * (isi_threshold_s - min_isi_s)
        )
    )


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(
    2,
    5,
    figsize=(18, 8),
    sharex=True,
    sharey=True,
)

axes = axes.flatten()

for n_violations, ax in zip(range(1, 11), axes):

    rp_contamination = compute_rp_contamination(
        n_violations=n_violations,
        n_spikes=spike_counts,
        recording_duration_s=RECORDING_DURATION_S,
        refractory_period_s=rp_s,
        censored_period_s=censored_s,
    )

    isi_ratio = compute_isi_violations_ratio(
        n_violations=n_violations,
        n_spikes=spike_counts,
        recording_duration_s=RECORDING_DURATION_S,
        isi_threshold_s=isi_threshold_s,
        min_isi_s=min_isi_s,
    )

    ax.plot(
        spike_counts,
        rp_contamination,
        alpha=0.65,
        linewidth=2,
        label=f"RP contamination ({RP_MS:g} ms)",
    )

    ax.plot(
        spike_counts,
        isi_ratio,
        alpha=0.65,
        linewidth=2,
        label=f"ISI ratio ({ISI_THRESHOLD_MS:g} ms)",
    )

    ax.set_title(
        f"{n_violations} violation"
        + ("s" if n_violations > 1 else "")
    )

    ax.grid(alpha=0.25)

    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, 0.2))  


# Shared labels
fig.supxlabel("Number of spikes")
fig.supylabel("Metric value")

# One shared legend
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=False,
)

fig.suptitle(
    f"Effect of spike count on refractory-period metrics\n"
    f"Recording duration = {RECORDING_DURATION_S:.1f} s",
    y=1.02,
)

plt.tight_layout()
plt.show()