import matplotlib.pyplot as plt 
import seaborn as sns
from typing import Deque, List, Optional, Tuple
from pydantic import BaseModel

class PlotItem(BaseModel):
    """
    Data and its configuration to plot
    """
    ptype: str="plot"
    xlabel: str=""
    ylabel: str=""
    xlim: tuple=None 
    ylim: tuple=None
    title: str=""
    x: list=None
    y: list=None
    args: List[dict]=None

class PlotHelper(BaseModel):
    """
    configuration of in total plot
    """
    row: int=1
    col: int=1
    figsize: tuple=(10, 10)
    title: str="" 
    plots: List[PlotItem]=None


def draws(helper: PlotHelper):
    fig, axs = plt.subplots(helper.row, helper.col, figsize=helper.figsize)

    iteration = iter(helper.plots)

    if helper.row == 1 and helper.col == 1: # means only one plot
        plot_branch = {
            "plot": draw_plot,
            "hist": draw_hist,
            "bar": draw_bar,
            "heatmap": draw_heatmap,
        }
        
        target = next(iteration) # choose data
        plot_branch[target.ptype](axs, target) # choose accordingly relying on plot type
        
        axs.set(xlabel=target.xlabel, ylabel=target.ylabel)
        axs.set_xlim(target.xlim) if target.xlim is not None else None
        axs.set_ylim(target.ylim) if target.ylim is not None else None
        axs.set_title(target.title)
        axs.legend()


    else:
        for ax, target in zip(axs.flat, helper.plots):
            plot_branch = {
                "plot": draw_plot,
                "hist": draw_hist,
                "bar": draw_bar,
                "heatmap": draw_heatmap,
            }
            target = next(iteration) # choose data
            plot_branch[target.ptype](ax, target) # choose accordingly relying on plot type
            
            ax.set_xlim(target.xlim) if target.xlim is not None else None
            ax.set_ylim(target.ylim) if target.ylim is not None else None
            ax.set_title(target.title) 
            ax.legend()

            ax.set(xlabel=target.xlabel, ylabel=target.ylabel)

        fig.suptitle(helper.title)

def draw_plot(ax, target):
    for x, y, args in zip(target.x, target.y, target.args):
        ax.plot(x, y, **args)

def draw_hist(ax, target):
    for data, args in zip(target.x, target.args):
        ax.hist(x=data, **args)        

def draw_bar(ax, target):
    for x, y, args in zip(target.x, target.y, target.args):
        ax.bar(x, y, **args)

def draw_heatmap(ax, target):
    for data, args in zip(target.x, target.args):
        sns.heatmap(data=data, ax=ax, **args)