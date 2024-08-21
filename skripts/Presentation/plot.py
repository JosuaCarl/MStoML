"""
Methods for plotting.
"""

# Imports
import os

import regex

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from significance_analysis import *



def add_methodology(df, dim_red_method:str, classification_algorithm:str):
    df["Dimensionality reduction method"] = pd.Series([dim_red_method]*len(df))
    df["Classification algorithm"] = pd.Series([classification_algorithm]*len(df))
    return df


def str_to_np_ndarray(bracketed_string:str, sep:str) -> np.ndarray:
    bracketed_string = bracketed_string[1:-1]
    if "[" in bracketed_string:
        split = regex.findall(r'\[(?:[^][]|(?R))*\]', bracketed_string)
        return np.array([str_to_np_ndarray(bracketed_string, sep) for bracketed_string in split])
    else:
        return np.fromstring(bracketed_string, sep=sep)
    

def collect_all_dr_methods(dim_red_methods:dict,  algorithms:list, metrics_level:str, source:str, start:str):
    dr_dfs = []
    for dim_red_method, recon_loss in dim_red_methods.items():
        project = source if "nnot" in dim_red_method else f"{source}_{recon_loss}"
        dr_category = "annot" if "nnot" in dim_red_method else "latent"
        for algorithm in algorithms:
            path = os.path.join(f"{start}/runs/ML/", dr_category, project, f"{algorithm}_{metrics_level}.tsv")
            algorithm = algorithm.replace("SK-learn", "")
            algorithm = algorithm.replace(" RF", "")
            df = add_methodology( pd.read_csv( path, sep="\t", index_col=0),
                                        dim_red_method=dim_red_method, classification_algorithm=algorithm)
            df["TPR"] = df["TPR"].apply(lambda x: str_to_np_ndarray(x, " "))
            df["FPR"] = df["FPR"].apply(lambda x: str_to_np_ndarray(x, " "))
            df["Threshold"] = df["Threshold"].apply(lambda s: s.replace("inf", "1."))
            df["Threshold"] = df["Threshold"].apply(lambda x: str_to_np_ndarray(x, " "))
            df["Conf_Mat"] = df["Conf_Mat"].apply(lambda x: str_to_np_ndarray(x, " "))
            dr_dfs.append(df)
    dr_dfs = pd.concat( dr_dfs )
    return dr_dfs


def plot_dimred_results(df:pd.DataFrame, x, y, hue, source:str, outpath, plottype:str="violin", suffix:str=""):
    hue_order = list(df.groupby(hue)[y].mean().sort_values(ascending=False).index)
    sns.reset_defaults()
    if plottype=="cat":
        sns.set_style("darkgrid")
        ax = sns.catplot(
            data=df, x=x, y=y, hue=hue, hue_order=hue_order,
            kind="point", alpha=0.85, errorbar="se", aspect=1.7,
            palette=sns.blend_palette(["crimson", "gold", "violet", "turquoise", "mediumblue"], n_colors=len(hue_order), as_cmap=False),
            err_kws={"linewidth": 2.0, "alpha": 0.5},
            markersize=5
            #palette=sns.diverging_palette(30, 240, l=65, sep=1, center="dark", as_cmap=False, n=9)
        )
        ax.tick_params(axis='x', labelrotation=0, labelsize=9)
        plt.ylim((0.46, 0.87))
        plt.xlabel("VAE loss function")
        plt.title(f"Classifier performance on latent spaces", fontdict= { 'fontsize': 11, 'fontweight':'bold'})
        sns.move_legend(ax, "upper right", bbox_to_anchor=(0.75, 0.9775), ncol=3, frameon=True, title=None)
        #plt.legend(loc='upper right', title=hue, prop={'size': 8})
    if plottype=="box":
        sns.set_style("darkgrid")
        sns.set(rc={"figure.figsize":(10, 6)})
        hue = x
        ax = sns.boxplot(
            data=df, x=x, y=y, hue=hue,
            palette="vlag"
        )
        ax.tick_params(axis='x', labelrotation=0, labelsize=10)
    if plottype=="violin":
        sns.set_style("darkgrid")
        sns.set(rc={"figure.figsize":(10, 6)})
        hue = x
        ax = sns.violinplot(
            data=df, x=x, y=y, hue=hue,
            palette="vlag", cut=0, inner="box"
        )
        ax.tick_params(axis='x', labelrotation=0, labelsize=10)
        plt.title(f"Overall classification performance", fontdict= { 'fontsize': 11, 'fontweight':'bold'})
    plt.savefig(os.path.join(outpath, f"{plottype}_{source}_{x}_{y}_{hue}{suffix}.png"), bbox_inches="tight", dpi=600)
    plt.show()


def plot_auc(dim_red_methods, algorithms, metrics_level, source, group, group_ids, hue, outpath, suffix="", start:str="../.."):
    plt.close()
    df = collect_all_dr_methods(dim_red_methods=dim_red_methods, algorithms=algorithms, metrics_level=metrics_level, source=source, start=start)
    plt_df = pd.concat( [df[df[group] == group_id] for group_id in group_ids] ).reset_index(drop=True)

    if len(group_ids) > 1:
        for i, row in plt_df.iterrows():
            plt_df.loc[i, hue] = f"{row[group]} - {row[hue]}"


    ax = sns.lineplot( plt_df.explode(["TPR", "FPR"]), x="FPR", y="TPR", hue=hue, alpha=0.5)
    ax = sns.lineplot( pd.DataFrame({"TPR": [0., 1.], "FPR": [0., 1.], hue: ["Reference", "Reference"]}), x="FPR", y="TPR", hue=hue,
                    alpha=0.5, ax=ax, palette='Greys', linestyle='--')
    fig = ax.get_figure()

    legend = ax.axes.get_legend()
    legend.set_title(f"{hue} (AUC)")
    whitespace_width = 4.875
    max_label_width = np.max([t.get_window_extent().x1 for t in legend.texts])
    for i, t in enumerate(legend.texts):
        text = t.get_text()
        diff_extend = max_label_width - t.get_window_extent().x1
        whitspace_fill_count = int(np.round(diff_extend / whitespace_width)) + 2
        row = plt_df[plt_df[hue] == text]
        auc = 0.5 if row.empty else np.round(np.mean(row["AUC"]), 3)
        text = f'{text}{" "*whitspace_fill_count}({auc})'
        t.set_text(text)

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    if len(group_ids) > 1:
        title_text = " + ".join(group_ids)
    else :
        title_text = group_ids[0]

    if len(dim_red_methods) == 1:
        title_text += f" - {list(dim_red_methods.keys())[0]}"

         
    plt.title(f"ROC curve - {title_text}")
    plt.savefig(os.path.join(outpath, f"AUC_{source}_{hue}_{group}_{title_text}{suffix}.png"), bbox_inches="tight", dpi=600)
    plt.show()