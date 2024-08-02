from random import uniform
from scipy.stats import norm #type: ignore
import copy
import numpy as np
from ODE_tools import *
from RAS_ODE_models_MCMC import *
from multiprocessing import Pool
import mpl_scatter_density # adds projection='scatter_density' #type: ignore
import warnings
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
from scipy import stats #type: ignore
import matplotlib
import seaborn as sns
from rule_functions import *

CODE_VERSION = "1.0.0"

def sigmoid(x,n):
    return np.exp(x-n/2)/(1+np.exp(x-n/2))

def exp(x,n):
    return np.exp(x-n)

class Parameter:
    """Class sets up a data class to hold all relevent information about a given parameter."""

    def __init__(self,label:str,id:int|str,type:str,prior_mu:float,prior_std:float,q_std:float,bound_mults=[1e-2,1e2]):
        self.label = label          # optionally LATEX formatted string for labelling parameter on graphs.
        self.id = id                # string index to be used in data frames and dictionaries when saving data.
        self.prior_mu = prior_mu    # mean prior parameter value
        self.prior_std = prior_std  # prior standard deviation
        self.q_std = q_std          # standard deviation for proposal step.
        self.type = type            # kinetic or state. Is used for determining how to change parameters when running ODE.
        self.lb = self.prior_mu*bound_mults[0] # sets lower bound for proposal step.
        self.ub = self.prior_mu*bound_mults[1] # sets upper bound for proposal step

def func_wrapper(input):
    """function used as wrapper for running rule functions to be able to paralellize operation."""
    passed,val = input['rule_func'](input['o'],input['param_inds'],input['bounds'])
    return {'passed':passed,'val':val}
    
class MCMC:
    """Class MCMC handles all MCMC operations. Is set up with relevent parameter information as a list of Parameter objects."""

    def __init__(self,params:list[Parameter]):

        # parameter information
        self.params = params
        self.prior_lb = 0.5

        # rule information
        self.rule_funcs = rule_functions # from rule_functions.py
        self.upper_thresholds = [None,None,None,None,11,None,None]
        self.lower_thresholds = [0.09920443738680824*(0.5),100+0.5*(215.62967601111103-100),100+0.5*(215.62967601111103-100),50,None,0.5*29.292937875056282,0.5*59.439713008530475]     
        self.max_count_multiplier = 10
        self.liklihood_type = 'SIGMOID'
 
        # resulting data information
        self.burn_in = 0.2
        self.S = {}                 # parameter values from all accepted steps
        self.S_all = {}             # parameter values from all steps including not accepted.
        self.S_burned = {}          # parameter values from all accepted steps but burned.
        self.rules_passed_all = []  # lists of 1s or 0s marking which rules were accepted on each step including those not accepted.
        self.rule_passed_accepted = []  # same as rules_passed_all but for the accepted steps.
        self.vals_accepted = []         # lists of actual values of rules that were accepted on each step.
        self.P = {}                 # prior distribution
        self.prior_override = False # used to turn off any prior calculation.
        self.save = True            # used to turn on or off saving data to csv files.

    def q(self,o_old):
        """function takes last accepted parameters and calculates new ones based on guassian with mean as o_old and std as self.q_std."""
        
        new_o = {}
        max_tries = 1000

        # propose new parameter for each parameter in self.param_inds
        for param in self.params:
            flag = False
            new_o_i = -1
            count = 0
            while flag == False:
                count += 1

                # propose new log-normal parameter
                new_o_i = 10**np.random.normal(np.log10(o_old[param.id]),param.q_std)

                # check if within bounds
                if new_o_i > param.lb and new_o_i < param.ub:
                    flag = True

                # exit after max_tries if nothing found.
                if count > max_tries:
                    flag = True
                
            new_o[param.id] = new_o_i
        return new_o

    def test_gate(self,o,param_names):
        """function checks the gate rule from rule_functions"""
        flag, val = rule_initial_pass(o,param_names)

        return flag, val

    def test_proposed_parameter(self,o):
        """function tests proposed parameter against rules. Paralellized to evaluate each rule simultaneously."""

        p = 0
        counts = [0 for i in range(len(self.rule_funcs))]
        vals = [0 for i in range(len(self.rule_funcs))]

        flag, val = self.test_gate(o,self.params)

        if flag:
            n_rules = len(self.rule_funcs)

            # set up inputs for paralell processing Pool
            inputs = []
            for i in range(n_rules):
                rule_func = self.rule_funcs[i]
                inputs.append({'o':o,'param_inds':self.params,'rule_func':rule_func,'bounds':[self.lower_thresholds[i],self.upper_thresholds[i]]})

            # evaluate rules in paralell.
            pool=Pool(processes = n_rules)
            outputs = pool.map(func_wrapper,inputs)
            pool.close()
            pool.join()
    
            counts = [1 if output["passed"] else 0 for output in outputs]
            vals = [output['val'] for output in outputs]

            # determine probability based on likeihood function type.
            if self.liklihood_type == 'SIGMOID':
                p = sigmoid(np.sum(counts),n_rules)
            elif self.liklihood_type == 'EXP':
                p = exp(np.sum(counts),n_rules)
            elif self.liklihood_type == 'STRICT':
                if np.sum(counts) == len(counts):
                    p = 1
                else:
                    p = 0
            else:
                p = 0
                print("ERROR: ENTER VALID liklihood_type!")

        return p, counts, vals
    
    def prior_pdf(self,val,mean,std):
        """function checks individual proposed parameter value against prior dist."""

        if self.prior_override == True:
            p = 1
        else:
            p = norm.pdf(val,mean,std)
            if p < self.prior_lb:
                p = self.prior_lb
        return p
    
    def liklihood(self,o):
        """function finds likelihood of proposed parameter set"""

        # perform simulations returning P(D|proposed parameters)
        p_D_given_o, rules_passed, vals = self.test_proposed_parameter(o)

        # check prior probabilities of each proposed parameter and multiply them
        p_o_total = 1
        for param in self.params:
            p_o_total *= self.prior_pdf(np.log10(o[param.id]),np.log10(param.prior_mu),param.prior_std)

        # P(o|D) ~= P(D|o)*P(o)
        p_post = p_o_total*p_D_given_o

        return p_post, rules_passed, vals
    
    def test_new_o(self,o,likli_old):
        """performs Metropolis-Hasting algorithm to determine if proposed parameter is accepted."""

        # get new likelihood of proposed parameters
        likli_new,rules_passed,vals = self.liklihood(o)
        passed = False

        # avoid division by zero
        if likli_old == 0:
            likli_old = 1e-15

        # Metropolis-Hastings criterion
        if likli_new/likli_old > uniform(0,1):
            passed = True
            
        return passed,likli_new,rules_passed,vals
    
    def run_MCMC(self,n:int,o_0:dict,id=''):
        """runs MCMC to n*(1+self.burn_in) iterations using o_0 as initial dict of parameter values. id affects name of saved data. """

        S = {param.id: [] for param in self.params}                          # list of sample o's
        S_all = {param.id: [] for param in self.params}                      # includes repeated o's.
        o = o_0
        likli_old,rules_passed,vals = self.liklihood(o)

        sim_len = round(n*(1+self.burn_in))
        t = tqdm(range(sim_len),desc='Running MCMC...',miniters=int(sim_len/1000))
        count = 0
        max_count = sim_len*self.max_count_multiplier
        rules_passed_accepted = []
        rules_passed_all = []
        vals_accepted = []

        save_interval = int(n/10)
        interval_count = 0

        while count < round(n*(1+self.burn_in)):

            if count > max_count:
                print("Error - exceeded max_count.")
                break

            flag = True
            while flag:            
                
                o_proposed = o
                o_proposed = self.q(o_proposed) # propose new o
                passed,likli_new,rules_passed,vals = self.test_new_o(o_proposed,likli_old) # test new o
                rules_passed_all.append(rules_passed)

                if passed:
                    t.update()

                    # Update MCMC
                    likli_old = likli_new
                    o = o_proposed

                    # Update sample data and save on intervals
                    for param_new in self.params:
                        S_all[param_new.id].append(o[param_new.id])
                        S[param_new.id].append(o[param_new.id])
                    flag = False
                    count += 1
                    interval_count += 1
                    if interval_count >= save_interval and self.save:
                        self.save_data(id,count,S,S_all,rules_passed_all,rules_passed_accepted,vals_accepted)
                        interval_count = 0
                    rules_passed_accepted.append(rules_passed)
                    vals_accepted.append(vals)

                else:
                    for param_new in self.params:
                        S_all[param_new.id].append(o[param_new.id])

        self.S = S
        self.S_all = S_all
        self.rules_passed_all = rules_passed_all
        self.rules_passed_accepted = rules_passed_accepted
        self.vals_accepted = vals_accepted

        burn_in_len = round(self.burn_in*len(S[self.params[0].id]))
        S_burned = {}
        for param in self.params:
            S_burned[param.id] = S[param.id][burn_in_len:]
        self.S_burned = S_burned

        P = {}
        for param in self.params:
            P[param.id] = self.get_prior_dist(n,param)
        self.P = P

        return S, S_all, rules_passed_all, rules_passed_accepted, vals_accepted
    
    def get_prior_dist(self,n,param):
        "gets n samples of given param"
        P = np.power(10,np.random.normal(np.log10(param.prior_mu),param.prior_std,size=n))
        return P
    
    def save_data(self,id,n,S,S_all,rules_passed_all,rules_passed_accepted,vals_accepted):
        """saves all data to data folder. Names modified with id."""

        df = pd.DataFrame.from_dict(S)
        df.to_csv('runs/{}/data/S_{}.csv'.format(id,id),',')

        df = pd.DataFrame.from_dict(S_all)
        df.to_csv('runs/{}/data/S_all_{}.csv'.format(id,id),',')

        df = pd.DataFrame.from_dict(self.P)
        df.to_csv('runs/{}/data/P_{}.csv'.format(id,id),',')

        rules_passed_all = np.array(rules_passed_all)
        rules_passed_all.tofile('runs/{}/data/rules_passed_all_{}.csv'.format(id,id), sep = ',')
    
        rules_passed_accepted = np.array(rules_passed_accepted)
        rules_passed_accepted.tofile('runs/{}/data/rules_passed_accepted_{}.csv'.format(id,id), sep = ',')

        vals_accepted = np.array(vals_accepted)
        vals_accepted.tofile('runs/{}/data/vals_accepted_{}.csv'.format(id,id), sep = ',')

        #print('\nSaved data at {} iterations.\n'.format(n))

        return True
    
    def spider_plots(self, o_0, include_region=True, y_log_scale = True, path=None):
    
        multipliers = np.logspace(-2,2)
        y_log_scale = True

        for i,rule_func in enumerate(self.rule_funcs):
            plt.figure()
            fig, ax1 = plt.subplots()
            
            for param in self.params:
                vals = []
                for mult in multipliers:
                    o = copy.deepcopy(o_0)
                    o[param.id] = o[param.id]*mult
                    
                    passed,val = rule_func(o,self.params,[self.lower_thresholds[i],self.upper_thresholds[i]])
                    vals.append(val)
                if np.max(np.array(vals)) > 1000:
                    flag = True
                ax1.semilogx(multipliers,vals,label=param.label)
            #ax.legend(param_inds)
            ax1.set_title(rule_func.__name__)
            if y_log_scale:
                ax1.semilogy()
            if include_region:

                if self.lower_thresholds[i]:
                    ax1.axhline(self.lower_thresholds[i],c='green',linestyle='--',linewidth=1,label='{}%'.format(round(self.lower_thresholds[i]),3))
                if self.upper_thresholds[i]:
                    ax1.axhline(self.upper_thresholds[i],c='green',linestyle='--',linewidth=1,label='{}%'.format(round(self.upper_thresholds[i]),3))
                
                if self.lower_thresholds[i] and self.upper_thresholds[i]:
                    ax1.fill_between(ax1.get_xlim(),self.lower_thresholds[i],self.upper_thresholds[i], alpha=0.1,color='green')
                else:
                    y_min,y_max = ax1.get_ylim()
                    if self.lower_thresholds[i]:
                        ax1.axhline(y_max,c='green',linestyle='--',linewidth=1)
                        ax1.fill_between(ax1.get_xlim(),self.lower_thresholds[i],y_max, alpha=0.1,color='green')
                        ax1.set_ylim(y_min,y_max)
                    if self.upper_thresholds[i]:
                        ax1.axhline(y_min,c='green',linestyle='--',linewidth=1)
                        ax1.fill_between(ax1.get_xlim(),self.upper_thresholds[i],y_min, alpha=0.1,color='green')
                        ax1.set_ylim(y_min,y_max)        
                
            ax1.set_xlim(np.min(multipliers),np.max(multipliers))
            ax1.set_xlabel('parameter multiplier')
            ax1.set_ylabel('rule value')
            labelLines()
            if path:
                plt.savefig(f"{path}/{rule_func.__name__}_spiderplot.svg")
            else:
                plt.show()

    def rule_pairplot(self,logscale=True,path=None,regression=True):

        vals = {}
        rule_inds = []
        for i,rule in enumerate(self.rule_funcs):
            rule_inds.append(rule.__name__)
            vals[rule.__name__] = np.array(self.vals_accepted)[:,i]

        path_modified = path
        if path:
            path_modified = f"{path}/rule_pairplot.svg"

        rule_params = [Parameter(rule_func.__name__,rule_func.__name__,'rule_func',0,0,0) for rule_func in self.rule_funcs] # gross way to make pairplots work with rule funcs
        axs1 = pairplot(vals,rule_params,logscale=logscale,path=path_modified)

    def rule_percentage_heatmap(self,path=None):
        plt.figure()
        rules = np.array(self.rules_passed_accepted)
        rule_percentages = []
        n_steps = len(rules[:,0]) #type: ignore
        
        for i,primary_rule in enumerate(self.rule_funcs):
            rule_per_temp = []
            for j,other_rule in enumerate(self.rule_funcs):
                prim_passed = 0
                other_passed = 0
                for step in range(n_steps):
                    if rules[step,i]: #type: ignore
                        prim_passed += 1
                        if rules[step,j]: #type: ignore
                            other_passed += 1
                rule_per_temp.append(other_passed/prim_passed*100)
            rule_percentages.append(rule_per_temp)

        rule_inds = [rule.__name__ for rule in self.rule_funcs]
        percentage_df = pd.DataFrame(rule_percentages,index=rule_inds,columns=rule_inds)
        sns.clustermap(percentage_df,cmap='viridis')
        if path:
            plt.savefig(f"{path}/heat_map.svg")
        else:
            plt.show()

    def trace_plot(self,path=None):
        plt.figure()
        for param in self.params:
            plt.semilogy(self.S[param.id],linewidth=0.5,label=param.label)

        plt.xlabel('successfull MCMC iteration')
        plt.ylabel('parameter value') #param_labels[0])
        x=labelLines()
        #plt.legend(self.param_labels)
        if path:
            plt.savefig(f"{path}/trace_plot.svg")
        else:
            plt.show()

    def posterior_vs_prior(self,path=None):

        for param in self.params:
            plt.figure()
            sns.kdeplot(data=self.P, x=param.id, log_scale = True,fill=True,color='blue')#, stat='density')
            sns.kdeplot(data=self.S_burned, x=param.id, log_scale = True,fill=True,color='purple')#, stat='density')
            #plt.ylim([0,4])
            plt.legend(['prior','posterior'])
            plt.xlabel(param.label)
            if path:
                plt.savefig(f"{path}/{param.id}_post_vs_prior.svg")
            else:
                plt.show()

    def parameter_pairplot(self,path=None,regression=True,include_prior=False):

        path_modified = path
        if path:
            path_modified = f"{path}/parameter_pairplot.svg"

        axs = pairplot(self.S_burned,self.params,path=path_modified,P=self.P,p_plateau=self.prior_lb,prior_override=self.prior_override) #type: ignore

white_viridis = LinearSegmentedColormap.from_list('white_viridis',[(0,(1,1,1,1))]+[(i/256,matplotlib.colormaps['viridis'](i/256)) for i in range(1,257)], N=256)

def pairplot(data,params:list[Parameter],logscale=True,ignore_upper_tri=True,regression=True,legend=None,P=0,p_plateau=None,prior_override=False,title_text=None,path=None):
    n_params = len(params)
    fig, axs = plt.subplots(n_params,n_params,figsize=(40,40),subplot_kw={'projection': 'scatter_density'}) #,layout='constrained')

    warnings.filterwarnings("ignore")

    for i,param_i in enumerate(params):
        i_lims = [min(data[param_i.id]),max(data[param_i.id])]
        for j,param_j in enumerate(params):
            j_lims = [min(data[param_j.id]),max(data[param_j.id])]
            plt.sca(axs[i][j])

            if i==j and ignore_upper_tri: # plot posterior
    
                sns.kdeplot(data=data, x=param_i.id, log_scale = logscale,fill=True,color='purple')#, stat='density')
                xmin, xmax = axs[i][j].get_xlim()
                #plt.ylim([0,4])

                if P != 0:
                    if not p_plateau:
                        print("error - please provide p_plateau, set at 1")
                        p_plateau = 1

                    plt.figure()
                    ax = sns.kdeplot(P[param_i.id],log_scale= True) #type: ignore
                    x = np.array(ax.get_lines()[0].get_xdata())
                    y = np.array(ax.get_lines()[0].get_ydata())
                    plt.close()
                    
                    plt.sca(axs[i][j])
                    
                    for i_y in range(len(y)):
                        if prior_override:
                            y[i_y] = p_plateau
                        else:
                            if y[i_y] < p_plateau:
                                y[i_y] = p_plateau
                    
                    x = np.insert(x,0,xmin)
                    y = np.insert(y,0,p_plateau)
                    x = np.insert(x,len(x),xmax)
                    y = np.insert(y,len(y),p_plateau)

                    color = 'blue'
                    plt.semilogx(x,y,c=color,linestyle='--',linewidth=1) #type: ignore
                    #plt.text(np.min(x),p_plateau+0.1,f'prior plateau = {p_plateau}',c=color)
                    plt.fill_between(x,np.zeros(len(y)),y,alpha=0.1) #type: ignore
                    plt.legend(["posterior pdf","__nolabel__","prior pdf"])
                plt.xlabel(param_i.label)
                plt.xlim(i_lims)

            elif j > i: # skip upper triangle
                axs[i][j].axis('off')

            else:   # plot scatter
                x = data[param_i.id]
                y = data[param_j.id]

                axs[i][j].scatter_density(x, y, cmap=white_viridis)
                plt.xlabel(param_i.label)
                plt.ylabel(param_j.label)

                if regression:
                    
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
                    ymin, ymax = axs[i][j].get_ylim()
                    xmin, xmax = axs[i][j].get_xlim()
                    line_x = np.linspace(xmin,xmax,50)
                    line_y = np.array([slope*x_i + intercept for x_i in line_x])
                    plt.plot(line_x,line_y,linestyle='--',linewidth=1,c='r')
                    move = 0.03
                    x_text = xmin
                    y_text = ymin

                    x_scale = (xmax-xmin)
                    y_scale = (ymax-ymin)
                    
                    if np.min(x) > 0 and logscale:
                        x_text = np.sqrt(xmax*xmin)
                    else:
                        x_text = (xmax+xmin)/2
                    if np.min(y) > 0 and logscale:
                        y_text = x_text*slope+intercept
                    else:
                        y_text = x_text*slope+intercept

                    plt.text(x_text,y_text,f"$R^{2}=${round(r_value**2,3)}",path_effects = [path_effects.withStroke(linewidth=2, foreground='w')],c='r')
                plt.xlim(i_lims)
                plt.ylim(j_lims)
                if logscale:
                    if np.min(x) > 0:
                        plt.semilogx()
                    if np.min(y) > 0:
                        plt.semilogy()
    if title_text:
        fig.suptitle(title_text)
    if path:
        plt.savefig(path)
    else:
        plt.show()
    return axs