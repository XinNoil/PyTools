# import matplotlib.colors as mcolors

# def sorted_names(colors):
#     by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),
#                         name)
#                     for name, color in colors.items())
#     return [name for hsv, name in by_hsv]

# colors_maps = {'tableau':mcolors.TABLEAU_COLORS,'css':mcolors.CSS4_COLORS}
# colors_names = {'base':['k','g','r','b','c','y','m','grey','brown','orange','olive','purple','pink']}
# for color_type in colors_maps:
#     colors_names[color_type] = sorted_names(colors_maps[color_type])
import seaborn as sns
colors_names = {'base': ['k', 'b', 'r', 'g', 'c', 'y', 'm', 'grey', 'brown', 'orange', 'olive', 'purple', 'pink'], 'tableau': ['tab:gray', 'tab:brown', 'tab:orange', 'tab:olive', 'tab:green', 'tab:cyan', 'tab:blue', 'tab:purple', 'tab:pink', 'tab:red'], 'css': ['black', 'dimgray', 'dimgrey', 'gray', 'grey', 'darkgray', 'darkgrey', 'silver', 'lightgray', 'lightgrey', 'gainsboro', 'whitesmoke', 'white', 'snow', 'rosybrown', 'lightcoral', 'indianred', 'brown', 'firebrick', 'maroon', 'darkred', 'red', 'mistyrose', 'salmon', 'tomato', 'darksalmon', 'coral', 'orangered', 'lightsalmon', 'sienna', 'seashell', 'chocolate', 'saddlebrown', 'sandybrown', 'peachpuff', 'peru', 'linen', 'bisque', 'darkorange', 'burlywood', 'antiquewhite', 'tan', 'navajowhite', 'blanchedalmond', 'papayawhip', 'moccasin', 'orange', 'wheat', 'oldlace', 'floralwhite', 'darkgoldenrod', 'goldenrod', 'cornsilk', 'gold', 'lemonchiffon', 'khaki', 'palegoldenrod', 'darkkhaki', 'ivory', 'beige', 'lightyellow', 'lightgoldenrodyellow', 'olive', 'yellow', 'olivedrab', 'yellowgreen', 'darkolivegreen', 'greenyellow', 'chartreuse', 'lawngreen', 'honeydew', 'darkseagreen', 'palegreen', 'lightgreen', 'forestgreen', 'limegreen', 'darkgreen', 'green', 'lime', 'seagreen', 'mediumseagreen', 'springgreen', 'mintcream', 'mediumspringgreen', 'mediumaquamarine', 'aquamarine', 'turquoise', 'lightseagreen', 'mediumturquoise', 'azure', 'lightcyan', 'paleturquoise', 'darkslategray', 'darkslategrey', 'teal', 'darkcyan', 'aqua', 'cyan', 'darkturquoise', 'cadetblue', 'powderblue', 'lightblue', 'deepskyblue', 'skyblue', 'lightskyblue', 'steelblue', 'aliceblue', 'dodgerblue', 'lightslategray', 'lightslategrey', 'slategray', 'slategrey', 'lightsteelblue', 'cornflowerblue', 'royalblue', 'ghostwhite', 'lavender', 'midnightblue', 'navy', 'darkblue', 'mediumblue', 'blue', 'slateblue', 'darkslateblue', 'mediumslateblue', 'mediumpurple', 'rebeccapurple', 'blueviolet', 'indigo', 'darkorchid', 'darkviolet', 'mediumorchid', 'thistle', 'plum', 'violet', 'purple', 'darkmagenta', 'fuchsia', 'magenta', 'orchid', 'mediumvioletred', 'deeppink', 'hotpink', 'lavenderblush', 'palevioletred', 'crimson', 'pink', 'lightpink']}

def plot_fig(sns_func, df, xlabel=None, ylabel=None, xlim=None, ylim=None, 
             fig_param={}, 
             plot_params={}):
    sns.set(**{'style':'whitegrid', 'font_scale':1.6, 'font':'Times New Roman', **fig_param})
    g = sns_func(data=df, **plot_params)
    if xlabel is not None:
        if hasattr(g, 'set_xlabels'):
            g.set_xlabels(xlabel)
        else:
            g.set_xlabel(xlabel)
    if ylabel is not None:
        if hasattr(g, 'set_ylabels'):
            g.set_ylabels(ylabel)
        else:
            g.set_ylabel(ylabel)
    if xlim is not None:
        g.set(xlim=xlim)
    if ylim is not None:
        g.set(ylim=ylim)
    return g

def plot_cdf(df, x, xlabel=None, xlim=None, ylim=None, ylabel="CDF", 
             fig_param={}, 
             plot_params={}):
    return plot_fig(sns.displot, df, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, 
             fig_param={'style':'whitegrid', 'font_scale':1.6, 'font':'Times New Roman', **fig_param}, plot_params={'kind':'ecdf', 'x':x, **plot_params})