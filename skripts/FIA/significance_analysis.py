import sys
sys.path.append( '..' )

from helpers.normalization import *
from FIA import *
from ML.ML4com import *

from itertools import combinations

from scipy.stats import ttest_ind, mannwhitneyu, f_oneway
from statsmodels.sandbox.stats.multicomp import multipletests

import plotly.express as px


def test_metabolites_organism(data, community_composition, organism_idx, alpha, test):
    in_bool = [True if i==1 else False for i in community_composition.iloc[:,organism_idx].values]
    data_in = data[in_bool]
    data_notin = data[np.invert(in_bool)]

    statistic, p_values = test(data_in, data_notin)
    results = multipletests(p_values, alpha=alpha, method="bonferroni")
    
    fc = data_in.mean() / data_notin.mean()

    stats = pd.DataFrame({"sig":results[0], "p":results[1], "fc":fc.values,
                          "-log10p":-np.log10(results[1]), "log2fc":np.log2(fc.values)}).T
    stats.columns = data.columns
    return (pd.concat([data.copy(), stats]), in_bool)



def plot_volcanos(data, strains, community_composition, ref_masses,
                  color_map:dict, sig_p:float=None, sig_fc_pos:float=None, sig_fc_neg:float=None,
                  show_labels:bool=True, width:int=1600, height:int=900, outfolder:str="."):
    
    if not color_map:
        color_map = {"": "lightgrey",
            "significant + high positive fold-change": "red",
            "significant + high negative fold-change": "orange",
            "significant": "violet",
            "high positive fold change": "blue",
            "high negative fold change": "blue"}

    figures = {}
    test_stats = {}
    includes = {}
    for e, org in enumerate(strains["Organism"]):
        test_stat, included = test_metabolites_organism(data=data, community_composition=community_composition,
                                                        organism_idx=e, alpha=1e-21, test=f_oneway)

        colors = []
        labels = []
        text = []
        for idx, row in test_stat.T.iterrows():
            labels.append(str(idx))
            if sig_p and row["p"] <= sig_p:
                if  sig_fc_pos and row["fc"] >= sig_fc_pos:
                    colors.append("significant + high positive fold-change")
                    text.append(str(idx))

                elif sig_fc_neg and row["fc"] <= sig_fc_neg:
                    colors.append("significant + high negative fold-change")
                    text.append(str(idx))

                else:
                    colors.append("significant")
                    text.append("")
            
            else:
                if sig_fc_pos and row["fc"] >= sig_fc_pos:
                    colors.append("high positive fold change")
                    text.append("")


                elif sig_fc_neg and row["fc"] <= sig_fc_neg:
                    colors.append("high negative fold change")
                    text.append("")

                else:
                    colors.append("")
                    text.append("")

        df = pd.DataFrame()
        df["log2 fold-change"] = test_stat.loc["log2fc"]
        df["-log10 p-value"] =  test_stat.loc["-log10p"]
        df["fold-change"] =  test_stat.loc["fc"]
        df["p-value"] =  test_stat.loc["p"]
        df["ref_mass"] = ref_masses
        test_stat.loc["ref_mass"] = ref_masses
        text = text if show_labels else [""]*len(test_stat.columns)
        hover_data = ["p-value", "fold-change", "ref_mass"]
        title = str(org)
        figure = px.scatter(df,
                            x = "log2 fold-change",
                            y = "-log10 p-value",
                            title = title,
                            text = text,
                            hover_name = labels,
                            hover_data=hover_data,
                            color = colors,
                            color_discrete_map = color_map)
        if sig_fc_pos:
            figure.add_vline(x=np.log2(sig_fc_pos), line_width=1, line_dash="dash", line_color="grey")
        if sig_fc_neg:
            figure.add_vline(x=np.log2(sig_fc_neg), line_width=1, line_dash="dash", line_color="grey")
        if sig_p:
            figure.add_hline(y=-1 * np.log10(sig_p), line_width=1, line_dash="dash", line_color="grey")
        figure.update_traces(textposition='top center')
        figure.add_vline(x=0, line_width=1, line_color='black')
        figure.add_hline(y=0, line_width=1, line_color='black')

        figures[org] = figure
        test_stat.loc["position"] = colors
        test_stats[org] = test_stat
        includes[org] = included

        if not os.path.isdir(outfolder):
            os.mkdir(outfolder)
        figure.write_image(f"{outfolder}/volcano_{org}.png", width=width, height=height)
    return (figures, test_stats, includes)


def extract_metabolites_of_interest(test_stats):
    met_of_interest = {}
    p_vals = {}
    for org, ts in test_stats.items():
        query = ts.loc["position"].values == "significant + high positive fold-change"
        query += ts.loc["position"].values == "significant + high negative fold-change"
        include = [idx not in ["sig", "p", "fc", "-log10p", "log2fc", "position"] for idx in ts.index] 
        met_of_interest[org] = ts.loc[include, query]  
        p_vals[org] = ts.loc["p", query]        
    return met_of_interest, p_vals


def p_val_to_star(p:float):
    if p < 1e-4:
        return "****"
    elif p < 1e-3:
        return "***"
    elif p < 1e-2:
        return "**"
    elif p < 5e-2:
        return "*"
    else:
        return "ns"
    

def significance_plot(sig_df, includes, title, p_vals=None, ax=None, x_label_rot=90, x_labelsize=10, plottype="box", imp_df=None):
    """
    Plot the significance of 
    """
    fig = plt.figure()
    p_vals = p_vals if p_vals is not None else sig_df.loc["p"]
    
    plot_df = sig_df.copy()
    plot_df = plot_df.loc[[idx not in ["sig", "p", "fc", "-log10p", "log2fc", "position"] for idx in plot_df.index]]
    plot_df["included"] = includes
    plot_df = plot_df.melt(id_vars=["included"])
    if imp_df is not None:
        plot_df = plot_df.loc[[metName in imp_df["metNames"].unique() for metName in plot_df["metNames"].values]]

    if plottype == "box":
        ax = sns.boxplot(plot_df, ax=ax, x="metNames", y="value", hue="included")
    elif plottype == "violin":
        ax = sns.violinplot(plot_df, ax=ax, x="metNames", y="value", hue="included",
                            cut=0, inner="point", bw_adjust=0.5)
    else:
        raise(ValueError(f"{plottype} is not a valid plottype"))
    
    for i, met in enumerate(plot_df["metNames"].unique()):
        y = np.max(plot_df[plot_df["metNames"] == met]["value"])
        h = ax.get_ylim()[1] / 100
        y += h
        x1, x2 = (i -0.2, i+ 0.2)
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0, c="black")
        ax.text((x1 + x2)*0.5, y+ h, p_val_to_star(p_vals[met]), ha='center', va='bottom', color="black")

    ax.set(title=title)
    ax.tick_params(axis='x', rotation=x_label_rot, labelsize=x_labelsize)

    return (fig, ax)

def significance_plot_batch(targets_of_interest, includes, p_vals, x_label_rot:int=90,
                            x_labelsize=10, plottype="box", imp_df=None):
    """
    Returns:
        Figures of significance plots
    """
    plots = {}
    axs = {}
    for org, target_of_interest in tqdm(targets_of_interest.items()):
        plots[org], axs[org] = significance_plot(target_of_interest, includes[org], org, p_vals[org],
                                       x_label_rot=x_label_rot, x_labelsize=x_labelsize, plottype=plottype,
                                       imp_df=imp_df)
    return (plots, axs)


def importance_plot(imp_df, title, importance_cutoff=0.05, ax=None, x_label_rot=90, x_labelsize=10):
    """
    Plot the feature importance
    """
    fig = plt.figure()
    if isinstance(importance_cutoff, int):
        importance_cutoff = imp_df.sort_values(title, ascending=False)[title].to_numpy()[importance_cutoff]
    imp_df = imp_df.loc[imp_df[title] > importance_cutoff , [title, "metNames"]]
    imp_df.rename(columns={title: "feature importance"}, inplace=True)

    ax = sns.barplot(imp_df, ax=ax, x="metNames", y="feature importance", alpha=0.3, width=0.9, palette=["violet"])
    ax.set_ylim(bottom=0.0, top=1.0)
    ax.grid(False)
    ax.set(title=title)
    ax.tick_params(axis='x', rotation=x_label_rot, labelsize=x_labelsize)

    return (fig, ax, imp_df)

def importance_plot_batch(targets_of_interest, importances_df, importance_cutoff:float=0.05):
    """
    Returns:
        Figures of importance plots
    """
    plots = {}
    axs = {}
    for org, target_of_interest in targets_of_interest.items():
        plots[org], axs[org] = importance_plot(importances_df, org, importance_cutoff)
    return (plots, axs)



def significance_importance_plot(sig_df, imp_df, includes, title, p_vals, importance_cutoff=0.05, 
                                 x_label_rot=90, x_labelsize=8, plottype="box"):
    
    fig, ax = significance_plot(sig_df, includes, title, p_vals,
                                x_label_rot=x_label_rot, x_labelsize=x_labelsize, plottype=plottype)

    ax_twin = ax.twinx()
    fig, ax, imp_df = importance_plot(imp_df, title, importance_cutoff, ax=ax_twin,
                                      x_label_rot=x_label_rot, x_labelsize=x_labelsize)
    return (fig, ax)

def significance_importance_plot_batch(targets_of_interest, importances_df, includes, p_vals,
                                       importance_cutoff=0.05, 
                                       x_label_rot=90, x_labelsize=8, plottype="box"):
    plots = {}
    axs = {}
    for org, target_of_interest in tqdm(targets_of_interest.items()):
        plots[org], axs[org] = significance_importance_plot(target_of_interest, importances_df, includes[org], org, p_vals[org],
                                                  importance_cutoff=importance_cutoff,
                                                  x_label_rot=x_label_rot, x_labelsize=x_labelsize, plottype=plottype)
    return (plots, axs)


def importance_significance_plot(sig_df, imp_df, includes, title, ax=None, importance_cutoff:float=0.05,
                                 x_label_rot=90, x_labelsize=8, plottype="violin"):

    fig, ax, imp_df = importance_plot(imp_df, title, importance_cutoff,
                                      x_label_rot=x_label_rot, x_labelsize=x_labelsize, ax=ax)
    ax.grid(False)

    ax_twin = ax.twinx()
    fig, ax = significance_plot(sig_df, includes, title, p_vals=None, ax=ax_twin,
                                x_label_rot=x_label_rot, x_labelsize=x_labelsize, plottype=plottype,
                                imp_df=imp_df)

    return (fig, ax)

def importance_significance_plot_batch(targets_of_interest, importances_df, includes, axes=None, importance_cutoff=0.05,
                                       x_label_rot=90, x_labelsize=8, plottype="violin"):
    plots = {}
    axs = {}
    for i, (org, target_of_interest) in enumerate(tqdm(targets_of_interest.items())):
        if axes is not None:
            ax = axes.flatten()[i]
        plots[org], axs[org] = importance_significance_plot(target_of_interest, importances_df, includes[org], org,
                                                            importance_cutoff=importance_cutoff, x_labelsize=x_labelsize,
                                                            plottype=plottype, x_label_rot=x_label_rot, ax=ax)
    return (plots, axs)



def combine_organisms(com, organisms, p:int=2):
    organisms_2 = pd.DataFrame({"ID": list(combinations(organisms["ID"], p)),
                                "Organism": list(combinations(organisms["Organism"], p))})
    com_2 = pd.DataFrame(columns=organisms_2["ID"])
    for i, row in com.iterrows():
        in_list = [1 if row[c[0]] == 1.0 and row[c[1]] == 1.0 else 0 for c in com_2.columns]
        com_2.loc[i] = in_list
    com_2.columns = organisms_2["Organism"]
    return (com_2, organisms_2)