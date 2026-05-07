import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ALPHAS = [0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 1.0, 3.0]

DATA = {
    "SmolVLM2-2.2B": {
        0.01: {"decode_attn_KL": 1.888, "logit_KL": 5.228},
        0.03: {"decode_attn_KL": 1.682, "logit_KL": 5.710},
        0.05: {"decode_attn_KL": 1.545, "logit_KL": 6.050},
        0.07: {"decode_attn_KL": 1.440, "logit_KL": 5.924},
        0.10: {"decode_attn_KL": 1.301, "logit_KL": 6.399},
        0.30: {"decode_attn_KL": 0.709, "logit_KL": 4.325},
        1.00: {"decode_attn_KL": 0.000, "logit_KL": 0.000},
        3.00: {"decode_attn_KL": 0.507, "logit_KL": 3.405},
    },
    "Qwen3-VL-2B": {
        0.01: {"decode_attn_KL": 2.344, "logit_KL": 21.570},
        0.03: {"decode_attn_KL": 2.358, "logit_KL": 22.014},
        0.05: {"decode_attn_KL": 2.268, "logit_KL": 21.095},
        0.07: {"decode_attn_KL": 2.178, "logit_KL": 19.767},
        0.10: {"decode_attn_KL": 2.083, "logit_KL": 19.997},
        0.30: {"decode_attn_KL": 1.541, "logit_KL": 16.064},
        1.00: {"decode_attn_KL": 0.000, "logit_KL":  0.000},
        3.00: {"decode_attn_KL": 0.912, "logit_KL":  9.550},
    },
}

ATTN_COLOR  = "#4878CF"
LOGIT_COLOR = "#E87722"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

fig, axes = plt.subplots(1, 2, figsize=(9, 4))

for ax, (model, data) in zip(axes, DATA.items()):
    alphas = sorted(data.keys())
    attn_kl  = [data[a]["decode_attn_KL"] for a in alphas]
    logit_kl = [data[a]["logit_KL"]        for a in alphas]

    ax2 = ax.twinx()

    l1, = ax.plot(
        alphas, attn_kl,
        color=ATTN_COLOR, linewidth=1.5, marker="o", markersize=4,
        label="Decode attention KL",
    )
    l2, = ax2.plot(
        alphas, logit_kl,
        color=LOGIT_COLOR, linewidth=1.5, marker="s", markersize=4,
        label="Logit KL",
    )

    ax.axvline(1.0, color="#888888", linewidth=1.2, linestyle="--", zorder=0)
    ax.text(
        1.0, 0.98,
        "baseline\n(α = 1)",
        ha="center", va="top",
        fontsize=7.5, color="#666666",
        transform=ax.get_xaxis_transform(),
    )

    ax.set_xscale("log")
    ax.set_title(model)
    ax.set_xlabel(r"Visual token scale $\alpha$")
    ax.set_ylabel("Decode attention KL", color=ATTN_COLOR)
    ax2.set_ylabel("Logit KL", color=LOGIT_COLOR)
    ax.tick_params(axis="y", labelcolor=ATTN_COLOR)
    ax2.tick_params(axis="y", labelcolor=LOGIT_COLOR)

    ax.legend(handles=[l1, l2], frameon=False)

    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(5))

fig.tight_layout(pad=1.5)
fig.savefig("assets/exp1_dose_response.png", bbox_inches="tight", dpi=200)
print("saved assets/exp1_dose_response.png")
