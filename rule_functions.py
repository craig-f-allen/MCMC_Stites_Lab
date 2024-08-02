from ODE_tools import *
from RAS_ODE_models_MCMC import *

def get_get_param_opts(WT_new,o,params,GEF_mult,GAP_mult):
    get_param_opts = {'mutant':WT_new,'GTot':4e-7,'EffTot':4e-7,'GAP':(6e-11)*GAP_mult,'GTP':180e-6,'GDP':18e-6,'GEF':(2e-10)*GEF_mult,'fract_mut':1}
    for param in params:
        mult = 1
        if param.type == 'iv':
            if param.id == "GEF":
                mult = GEF_mult
            if param.id == "GAP":
                mult = GAP_mult
            get_param_opts[param.id] = o[param.id]*mult 
    return get_param_opts

# initial rule gate
def rule_initial_pass(o,params):

    WT_new = copy.deepcopy(WT)
    for param in params:
        if param.type == 'kinetic':
            setattr(WT_new,param.id,o[param.id])

    GEF_mult = 1
    GAP_mult = 1
    get_param_opts = get_get_param_opts(WT_new,o,params,GEF_mult,GAP_mult)
    sim = ODE_Simulation(RAS_model,get_params_RAS,get_param_opts)
    res = sim.integrate_model_to_ss()

    per_RAS_GTP_Tot = res["per_RAS_GTP_Tot"]*100

    if per_RAS_GTP_Tot >= 1 and per_RAS_GTP_Tot <= 10:
        pass
    else:
        return False, f'basal GEF, RAS:GTP, {per_RAS_GTP_Tot}'
    
    per_RAS_GTP_Eff = res["per_RAS_GTP_Eff"]*100
    
    if per_RAS_GTP_Eff > 1 and per_RAS_GTP_Eff < 10:
        pass
    else:
        return False, f'basal GEF, RAS:GTP:Eff, {per_RAS_GTP_Eff}'
    
    GEF_mult = 10
    GAP_mult = 1
    get_param_opts = get_get_param_opts(WT_new,o,params,GEF_mult,GAP_mult)
    sim = ODE_Simulation(RAS_model,get_params_RAS,get_param_opts)
    res_10xGEF = sim.integrate_model_to_ss()
    
    per_RAS_GTP_Tot_10xGEF = res_10xGEF["per_RAS_GTP_Tot"]*100

    if per_RAS_GTP_Tot_10xGEF >= 30:
        pass
    else:
        return False, f'10x GEF, RAS:GTP {per_RAS_GTP_Tot_10xGEF}'
    
    per_RAS_GTP_Eff_10xGEF = res_10xGEF["per_RAS_GTP_Eff"]*100
    
    if per_RAS_GTP_Eff_10xGEF/per_RAS_GTP_Eff >= 5 :
        pass
    else:
        return False, f'10x GEF, RAS:GTP:Eff {per_RAS_GTP_Eff_10xGEF/per_RAS_GTP_Eff}'
    
    return True, 'all cases passed.'

#=== ALL RULES MUST HAVE THE SAME DEFAULT ARGUMENTS AND RETURN TYPES. THIS IS WHAT MCMC_tools EXPECTS ===

def rule_hypersens(o,params,bounds,out_option='per_RAS_GTP_Eff'):

    WT_new = copy.deepcopy(WT)
    for param in params:
        if param.type == 'kinetic':
            setattr(WT_new,param.id,o[param.id])

    GEF_mult = 2
    GAP_mult = 1
    get_param_opts = get_get_param_opts(WT_new,o,params,GEF_mult,GAP_mult)
    sim = ODE_Simulation(RAS_model,get_params_RAS,get_param_opts)
    GEF_stim_basal_GAP = sim.integrate_model_to_ss()[out_option]

    GEF_mult = 1
    GAP_mult = 1
    get_param_opts = get_get_param_opts(WT_new,o,params,GEF_mult,GAP_mult)
    sim = ODE_Simulation(RAS_model,get_params_RAS,get_param_opts)
    basal_GEF_basal_GAP = sim.integrate_model_to_ss()[out_option]

    GEF_mult = 2
    GAP_mult = 0.5
    get_param_opts = get_get_param_opts(WT_new,o,params,GEF_mult,GAP_mult)
    sim = ODE_Simulation(RAS_model,get_params_RAS,get_param_opts)
    GEF_stim_reduced_GAP = sim.integrate_model_to_ss()[out_option]

    GEF_mult = 1
    GAP_mult = 0.5
    get_param_opts = get_get_param_opts(WT_new,o,params,GEF_mult,GAP_mult)
    sim = ODE_Simulation(RAS_model,get_params_RAS,get_param_opts)
    basal_GEF_reduced_GAP = sim.integrate_model_to_ss()[out_option]

    GEF_stim_delta = GEF_stim_basal_GAP-basal_GEF_basal_GAP
    GAP_red_delta = GEF_stim_reduced_GAP-basal_GEF_reduced_GAP

    sens = (GAP_red_delta)-(GEF_stim_delta)
    
    if sens >= bounds[0]: #0.5*0.09920443738680824:
        return True, sens
    else:
        return False, sens
    
def rule_half_NF1_KO(o,params,bounds,out_option='per_RAS_GTP_Eff'):

    WT_new = copy.deepcopy(WT)
    for param in params:
        if param.type == 'kinetic':
            setattr(WT_new,param.id,o[param.id])

    GAP_mult = 1
    get_param_opts = get_get_param_opts(WT_new,o,params,1,GAP_mult)
    sim = ODE_Simulation(RAS_model,get_params_RAS,get_param_opts)
    basal_GAP = sim.integrate_model_to_ss()[out_option]

    GAP_mult = 0.75
    get_param_opts = get_get_param_opts(WT_new,o,params,1,GAP_mult)
    sim = ODE_Simulation(RAS_model,get_params_RAS,get_param_opts)
    red_GAP = sim.integrate_model_to_ss()[out_option]

    try:
        half_NF1_KO_val = red_GAP/float(basal_GAP)*100
    except:
        half_NF1_KO_val = 100

    if half_NF1_KO_val > bounds[0]: #100+0.5*(215.62967601111103-100):
        return True, half_NF1_KO_val
    else:
        return False, half_NF1_KO_val
    
def rule_full_NF1_KO(o,params,bounds,out_option='per_RAS_GTP_Eff'):

    WT_new = copy.deepcopy(WT)
    for param in params:
        if param.type == 'kinetic':
            setattr(WT_new,param.id,o[param.id])

    GAP_mult = 0.75
    get_param_opts = get_get_param_opts(WT_new,o,params,1,GAP_mult)
    sim = ODE_Simulation(RAS_model,get_params_RAS,get_param_opts)
    basal_GAP = sim.integrate_model_to_ss()[out_option]

    GAP_mult = 0.5
    get_param_opts = get_get_param_opts(WT_new,o,params,1,GAP_mult)
    sim = ODE_Simulation(RAS_model,get_params_RAS,get_param_opts)
    red_GAP = sim.integrate_model_to_ss()[out_option]

    flag,half_NF1_KO_val = rule_half_NF1_KO(o,params,bounds,out_option=out_option)

    try:
        full_NF1_KO_val = red_GAP/float(basal_GAP)*100
    except:
        full_NF1_KO_val = 100

    # First check if full val greater than half val:
    if full_NF1_KO_val < half_NF1_KO_val:
        return False, full_NF1_KO_val

    # Then check threshold for full val:
    if full_NF1_KO_val > bounds[0]: #100+0.5*(215.62967601111103-100):
        return True,full_NF1_KO_val
    
    else:
        return False,full_NF1_KO_val
    
def rule_GAP_insens(o,params,bounds,out_option='per_RAS_GTP_Tot'):
    WT_new = copy.deepcopy(WT)
    for param in params:
        if param.type == 'kinetic':
            setattr(WT_new,param.id,o[param.id])
    
    kint = copy.deepcopy(WT_new.kint)
    setattr(WT_new,'kint',kint*0.2)
    setattr(WT_new,'kcat',kint*0.2)

    get_param_opts = get_get_param_opts(WT_new,o,params,1,1)
    sim = ODE_Simulation(RAS_model,get_params_RAS,get_param_opts)
    GAP_insens_val = sim.integrate_model_to_ss()[out_option]*100 # convert to percent

    if GAP_insens_val > bounds[0]: # 50:
        return True, GAP_insens_val
    else:
        return False, GAP_insens_val

def rule_GTPase_decrease(o,params,bounds,out_option='per_RAS_GTP_Tot'):
    WT_new = copy.deepcopy(WT)
    for param in params:
        if param.type == 'kinetic':
            setattr(WT_new,param.id,o[param.id])

    get_param_opts = get_get_param_opts(WT_new,o,params,1,1)
    sim = ODE_Simulation(RAS_model,get_params_RAS,get_param_opts)
    basal_val = sim.integrate_model_to_ss()[out_option]*100

    mult = 0.1 # decreases kint
    kint = copy.deepcopy(WT_new.kint)
    setattr(WT_new,'kint',kint*mult)

    get_param_opts = get_get_param_opts(WT_new,o,params,1,1)
    sim = ODE_Simulation(RAS_model,get_params_RAS,get_param_opts)
    GTPase_dec_val = sim.integrate_model_to_ss()[out_option]*100 # convert to percentage

    try:
        activation_ratio = GTPase_dec_val/basal_val
    except:
        activation_ratio = 1

    if GTPase_dec_val < bounds[1] and activation_ratio < 4:#GTPase_dec_val < 11 and activation_ratio < 4:
        return True, GTPase_dec_val
    else:
        return False, GTPase_dec_val
    
def rule_GTPase_dec_comb(o,params,bounds,out_option='per_RAS_GTP_Tot'):
    WT_new = copy.deepcopy(WT)
    for param in params:
        if param.type == 'kinetic':
            setattr(WT_new,param.id,o[param.id])

    kint = copy.deepcopy(WT_new.kint)
    setattr(WT_new,'kcat',kint)

    get_param_opts = get_get_param_opts(WT_new,o,params,1,1)
    sim = ODE_Simulation(RAS_model,get_params_RAS,get_param_opts)
    basal_kint = sim.integrate_model_to_ss()[out_option]*100

    mult = 0.1 # decreases kint
    kint = copy.deepcopy(WT_new.kint)
    setattr(WT_new,'kint',kint*mult)
    setattr(WT_new,'kcat',kint*mult)

    get_param_opts = get_get_param_opts(WT_new,o,params,1,1)
    sim = ODE_Simulation(RAS_model,get_params_RAS,get_param_opts)
    dec_kint = sim.integrate_model_to_ss()[out_option]*100

    delta = dec_kint - basal_kint

    if delta > bounds[0]: #0.5*29.292937875056282:
        return True, delta
    else:
        return False, delta

def rule_fast_cycling(o,params,bounds,out_option='per_RAS_GTP_Tot'):

    WT_new = copy.deepcopy(WT)
    for param in params:
        if param.type == 'kinetic':
            setattr(WT_new,param.id,o[param.id])

    get_param_opts = get_get_param_opts(WT_new,o,params,1,1)
    sim = ODE_Simulation(RAS_model,get_params_RAS,get_param_opts)
    basal = sim.integrate_model_to_ss()[out_option]*100

    # make fast cycling
    kdissD = copy.deepcopy(WT_new.kdissD)
    kdissT = copy.deepcopy(WT_new.kdissT)
    setattr(WT_new,'kdissD',kdissD*25)
    setattr(WT_new,'kdissT',kdissT*25)

    get_param_opts = get_get_param_opts(WT_new,o,params,1,1)
    sim = ODE_Simulation(RAS_model,get_params_RAS,get_param_opts)
    fast_cycling = sim.integrate_model_to_ss()[out_option]*100

    delta = fast_cycling - basal

    if delta > bounds[0]:#0.5*59.439713008530475:
        return True, delta
    else:
        return False, delta

""" on hold
def rule_EGFRi_KM(o,param_names,eta=0,out_option='per_RAS_GTP_Tot'):

    WT_new = copy.deepcopy(WT)
    for param_name in param_names:
        setattr(WT_new,param_name,o[param_name])

    Km_50 = copy.deepcopy(WT_new.Km)*50
    setattr(WT_new,'Km',Km_50)

    kint = copy.deepcopy(WT_new.kint)
    setattr(WT_new,'kcat',kint)

    fract_mut = 0.25

    GEF_mult = 10
    get_param_opts = {'mutant':WT_new,'GTot':4e-7,'EffTot':4e-7,'GAP':(6e-11),'GTP':180e-6,'GDP':18e-6,'GEF':(2e-10)*(GEF_mult),'fract_mut':fract_mut}
    sim = ODE_Simulation(RAS_model,get_params_RAS,get_param_opts)
    no_drug = sim.integrate_model_to_ss()[out_option]

    GEF_mult = 1
    get_param_opts = {'mutant':WT_new,'GTot':4e-7,'EffTot':4e-7,'GAP':(6e-11),'GTP':180e-6,'GDP':18e-6,'GEF':(2e-10)*GEF_mult,'fract_mut':fract_mut}
    sim = ODE_Simulation(RAS_model,get_params_RAS,get_param_opts)
    drug = sim.integrate_model_to_ss()[out_option]

    try:
        change = no_drug / drug*100
    except:
        change = 100
    if change > 120*(1+eta) and change < 220*(1-eta):
        return True, change
    else:
        return False, change
    
"""    

rule_functions = [rule_hypersens,rule_half_NF1_KO,rule_full_NF1_KO,rule_GAP_insens,rule_GTPase_decrease,rule_GTPase_dec_comb,rule_fast_cycling]