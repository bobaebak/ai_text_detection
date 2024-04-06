import matplotlib.pyplot as plt 
from typing import Deque, List, Optional, Tuple
from pydantic import BaseModel

class PlotItem(BaseModel):
    ptype: str="plot"
    xlabel: str=""
    ylabel: str=""
    title: str=""
    x: list=None
    y: list=None
    args: dict={}

class PlotHelper(BaseModel):
    row: int=1
    col: int=1
    figsize: tuple=(10, 10)
    title: str="" 
    plots: Optional[List[PlotItem]] = None


def draw_plots(plot_helper: PlotHelper):
    fig, axs = plt.subplots(plot_helper.row, plot_helper.col, figsize=plot_helper.figsize)

    idx = 0
    for r in range(plot_helper.row):
        for c in range(plot_helper.col):
            plot_type = {
                "plot": axs[r, c].plot,
                "hist": axs[r, c].hist,
            }
            
            plot_args_data = plot_helper.plots[idx]
            plot_type[plot_args_data.ptype](x=plot_args_data.x, **plot_args_data.args)
            axs[r, c].set_title(plot_args_data.title)

            idx += 1

    for ax, plot_args_data in zip(axs.flat, plot_helper.plots):
        ax.set(xlabel=plot_args_data.xlabel, ylabel=plot_args_data.ylabel)

    fig.suptitle(plot_helper.title)


