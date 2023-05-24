import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="white")


def sigmoid(x, scale=1, factor=1):
    return factor * np.exp(-np.logaddexp(0, -x / scale))


def log_sigmoid(x, scale=1, factor=1):
    return factor * np.sign(x) * np.log(1 + abs(x / scale)) / (1 + np.log(1 + abs(x / scale)))


def k_sigmoid(x, k=1, scale=1, factor=1):
    return factor * (np.sign(x) * abs(x / scale) ** k) / (1 + abs(x / scale))


def linear(x, scale=1):
    return x / scale


def compute_std(x, norm):
    return math.sqrt(x / norm)


def plot_pairgrid(data, columns_names=['O', 'E', 'I']):
    colors = [plt.cm.tab10.colors[i:i + 3] for i in range(0, 2, 2)]
    hatches = ['xx', '', '||']
    cols = [column for column in list(data.columns)][:-1]

    palettes = {columns_names[0]: (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),  # blu
                columns_names[1]: (1.0, 0.4980392156862745, 0.054901960784313725),  # orange
                columns_names[2]: (0.17254901960784313, 0.6274509803921569, 0.17254901960784313)}  # green

    g = sns.PairGrid(data, hue="Intervention", palette=palettes)
    g.map_lower(sns.histplot, bins=70, alpha=0.7, thresh=3)

    for i, col in enumerate(cols):
        for palette in colors:
            kde = sns.kdeplot(data=data, x=col, hue="Intervention", hue_order=columns_names, fill=True, palette=palette,
                            ax=g.axes[i, i])
            kde.legend_.remove()
        for collection, hatch in zip(kde.collections[::-1], hatches * len(columns_names)):
            collection.set_hatch(hatch)

    for i in range(g.axes.shape[0]):
        for j in range(g.axes.shape[1]):
            if i < j:
                g.axes[i, j].xaxis.set_ticks([])
                g.axes[i, j].yaxis.set_ticks([])

    g.axes[0, 0].tick_params(axis='both', which='both', length=0)

    g.fig.legend(
        labels=['Observational Distribution', 'Interventional Distribution of E', 'Interventional Distribution of I'],
        loc='upper right',
        fontsize='large')

    plt.plot()
    plt.savefig('method.png', bbox_inches='tight', dpi=200)
    return


def plot_variances(data1, data2):

    lw = 1.4
    lower_bound = -3
    upper_bound = -4
    lower_bound_1 = 4
    upper_bound_1 = 6.5

    sns.kdeplot(data1, fill=True, label='A')
    sns.kdeplot(data2, fill=True, label='B')

    plt.axvline(x=lower_bound, ymax=0.56, color='blue', linewidth=lw)
    plt.axvline(x=upper_bound, ymax=0.95, color='blue', linewidth=lw)
    plt.axvline(x=lower_bound_1, ymax=0.47, color='red', linewidth=lw)
    plt.axvline(x=upper_bound_1, ymax=0.21, color='red', linewidth=lw)

    plt.legend(loc="upper left")

    plt.xlim(-8, 11)

    plt.plot()
    return
