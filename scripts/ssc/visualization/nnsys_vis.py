import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":

    df_path = '/Users/simons/polybox/Studium/20FS/MT/analysis/shortcut_count.csv'

    df = pd.read_csv(df_path)

    g = sns.catplot(x="k", y="count",
                    hue="type", col="n_samples",
                    data=df, kind="bar",
                    height=4, aspect=.7);

    plt.show()