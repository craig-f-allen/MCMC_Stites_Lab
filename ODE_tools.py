import numpy as np
from scipy.integrate import solve_ivp #type: ignore
import matplotlib.pyplot as plt
import copy
from labellines import labelLine, labelLines
from tqdm import tqdm
import pandas as pd

class ODE_Simulation:
    """This class is used to setup, and run ODE simulations. It uses scipy.integrate.solve_ivp to integrate using the necessary function format. """
    
    def __init__(self,model_fun,get_param_fun,param_ivs:dict):

        self.model_fun = model_fun          # actual scipy.integrate.solve_ivp compliant ODE function.
        self.get_param_fun = get_param_fun  # function that takes param_ivs and returns the resulting y0 and k parameters.
        self.param_ivs = param_ivs          # dict with initial values of parameters to be used in calculting y0 and k through get_param_fun
        
        self.k,self.y0 = self.get_param_fun(param_ivs)
        self.t = []
        self.y = np.array([])

    def integrate_model(self,t_end,y0=None,k=None,plot_option=False):
        """Integrates model from t = 0 to t = t_end. Can provide custom y0 and k parameters, or it will run with the default self.y0 and self.k. plot_option = True will turn on automatic plotting. Results are saved to self.t and self.y"""
        
        if k is None:
            k = copy.deepcopy(self.k)
        if y0 is None:
            y0 = copy.deepcopy(self.y0)

        sol = solve_ivp(self.model_fun, [0,t_end], y0, args = (k,),method='LSODA',rtol=1e-6,atol=1e-11)
        self.t = np.transpose(sol['t'])
        self.y = np.array(sol['y'])
        
        if plot_option:
            plt.plot(self.t,np.transpose(self.y))
            plt.xlabel('t')
            plt.ylabel('y')

        return self.t,self.y
    
    def integrate_model_to_ss(self,k=None,y0=None,y0_original=None):
        """Itegrates model to steady state. Can provide custom y0 and k parameters, or it will run with the default self.y0 and self.k. y0_original is used for situations where you need to compare to your own custom y0_original, ignore unless needed. Function outputs a results dictionary which contains lists of various outputs."""

        t_max = 10000
        dmet=[1 for i in range(len(self.y0))]
        whileval = 1e-32
        if y0 is None:
            y0 = copy.deepcopy(self.y0)
        if y0_original is None:
            y0_original = copy.deepcopy(y0)

        while np.dot(dmet,dmet)>whileval:
            y0_old = y0
            t,y = self.integrate_model(t_max,k=k,y0=y0)
            y0=self.y[:,-1]
            dmet=(y0-y0_old)
        
        # results dictionary used to output whatever analyses are needed. Some defaults are provided but user can add their own.
        results = {}
        results['y_ss'] = self.y[:,-1]
        #results['signal'] = (self.y[1,-1]+self.y[4,-1]+self.y[8,-1]+self.y[6,-1])/(y0_original[1]+y0_original[4]+y0_original[8]+y0_original[6])*100
        results['total'] = self.y[1,-1]+self.y[4,-1]+self.y[8,-1]+self.y[6,-1]
        results['per_RAS_GTP_Tot'] = (self.y[1,-1]+self.y[4,-1]+self.y[6,-1]+self.y[8,-1])/(y0_original[0]+y0_original[5]) # percent RAS-GTP out of total RAS pool
        results['per_RAS_GTP_Eff'] = (self.y[4,-1]+self.y[8,-1])/(y0_original[3]) # percent RAS-GTP-Eff out of total Eff pool
        results['RAS_GTP_Eff'] = (self.y[4,-1]+self.y[8,-1]) # percent RAS-GTP-Eff out of total Eff pool
        if y0_original[0] > 0:
            results['per_WT_RAS_GTP'] = (self.y[1,-1]+self.y[4,-1])/(y0_original[0])
        else:
            results['per_WT_RAS_GTP'] = 0
        results['per_WT_RAS_GTP_Eff'] = (self.y[4,-1])/(y0_original[3])
        if y0_original[5] > 0:
            results['per_Mut_RAS_GTP'] = (self.y[6,-1]+self.y[8,-1])/(y0_original[5])
        else:
            results['per_Mut_RAS_GTP'] = 0
        results['per_Mut_RAS_GTP_Eff'] = (self.y[8,-1])/(y0_original[3])
        results['per_WT_RAS_GTP_Tot'] = (self.y[1,-1]+self.y[4,-1])/(y0_original[0]+y0_original[5])
        results['per_Mut_RAS_GTP_Tot'] = (self.y[6,-1]+self.y[8,-1])/(y0_original[0]+y0_original[5])
        
        return results

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="",title=None,show_values=False,colorbar=True,**kwargs):
    #src: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
    plt.figure(figsize=(10,10))

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    if colorbar:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw) #type: ignore
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on bottom.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-45, ha="right",
             rotation_mode="anchor")
    
    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1), minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1), minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    if show_values:
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                text = ax.text(j, i, data[i, j],
                            ha="center", va="center", color="w")
    if title is not None: plt.title(title)

    return im

