from MCMC_tools import * #type: ignore
import os
import shutil
import sys
import numpy as np
import pandas as pd
import csv

def load_settings(file_path='settings.csv'):
    settings = {}
    with open(file_path) as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            settings[row[0]] = row[1]
    return settings

if __name__ == "__main__":

    settings_path = sys.argv[1]
    settings = load_settings(file_path=settings_path)

    print("\nStarting MCMC run. Run name: {}\n".format(settings['id']))

    try:
        os.mkdir('runs')
    except FileExistsError:
        pass
    try:
        os.mkdir(f"runs/{settings['id']}")
    except FileExistsError:
        print(f"Folder with id {settings['id']} already exists in /runs/. Overwriting existing data.")
    try:
        os.mkdir(f"runs/{settings['id']}/data")
    except FileExistsError:
        pass
    shutil.copyfile(settings_path,f"runs/{settings['id']}/run_settings.csv")

    q_std = float(settings['q_std'])
    std = float(settings['prior_std'])

    kint = Parameter('$k_{hyd}$','kint','kinetic',3.5e-4,std,q_std)
    kdissD = Parameter('$k_{d,GDP}$','kdissD','kinetic',1.1e-4,std,q_std)
    kdissT = Parameter('$k_{d,GTP}$','kdissT','kinetic',2.5e-4,std,q_std)
    kassD = Parameter('$k_{a,GDP}$','kassD','kinetic',2.3e6,std,q_std)
    kcat = Parameter('$k_{cat,GAP}$','kcat','kinetic',5.4,std,q_std)
    Km = Parameter('$K_{M,GAP}$','Km','kinetic',.23e-6/250,std,q_std)
    kD = Parameter('$k_{cat,GDP}$','kD','kinetic',3.9,std,q_std)
    KmD = Parameter('$K_{M,GDP}$','KmD','kinetic',3.86e-4/250,std,q_std)
    KmT = Parameter('$K_{M,GTP}$','KmT','kinetic',3e-4/250,std,q_std)
    Kd = Parameter('$K_{D,Eff}$','Kd','kinetic',80e-9,std,q_std)

    Eff = Parameter('Effector','EffTot','iv',4e-7,0.03,0.03,bound_mults=[0.1,10])
    GAP = Parameter('GAP','GAP','iv',(6e-11),0.03,0.03,bound_mults=[0.1,10])
    GEF = Parameter('GEF','GEF','iv',(2e-10),0.03,0.03,bound_mults=[0.1,10])
    #fract_mut = Parameter('% Mutant','fract_mut','iv',0.25,0.01,0.01,bound_mults=[0.5,2])

    params = [kint,kdissD,kdissT,kassD,kcat,Km,kD,KmD,KmT,Kd] #,Eff,GAP,GEF]
    #params = [Eff,GAP,GEF]#,fract_mut]

    MCMC_test = MCMC(params)
    
    MCMC_test.liklihood_type = settings['likelihood_func']
    MCMC_test.prior_lb = float(settings['prior_plateau'])
    MCMC_test.burn_in = float(settings['burn_in'])
    MCMC_test.upper_thresholds = [None,None,None,None,11,None,None]
    MCMC_test.lower_thresholds = [0.09920443738680824*(0.5),100+0.5*(215.62967601111103-100),100+0.5*(215.62967601111103-100),50,None,0.5*29.292937875056282,0.5*59.439713008530475]     
    if settings['prior_override'] == '1':
        MCMC_test.prior_override = True
        print("Prior overridden.")
    else:
        MCMC_test.prior_override = False

    o_0 = {}
    for param in params:
        o_0[param.id] = param.prior_mu

    S, S_all,rules_passed_all,rules_passed_accepted,vals_accepted = MCMC_test.run_MCMC(int(settings['n']),o_0,id=settings['id'])
    
    MCMC_test.save_data(settings['id'],int(settings['n']),S,S_all,rules_passed_all,rules_passed_accepted,vals_accepted)

    print(f"MCMC run complete. Data saved to runs/{settings['id']}/data. Acceptance ratio: {round(len(S[params[0].id])/len(S_all[params[0].id]),4)}")

    if settings['analyze_data']:

        analysis_path = f"runs/{settings['id']}/analysis"
        print("\nAnalyzing data...")
        try:
            os.mkdir(analysis_path)
        except FileExistsError:
            print(f"Analysis folder in folder runs/{settings['id']} already exists. Overwriting existing data.")

        if settings['spider_plots']=='1':
            print("Generating spiderplots...")
            MCMC_test.spider_plots(o_0,path=analysis_path)

        print("Generating rule pairplots...")
        MCMC_test.rule_pairplot(path=analysis_path)

        print("Generating rule percentage heatmap...")
        MCMC_test.rule_percentage_heatmap(path=analysis_path)

        print("Generating trace plot...")
        MCMC_test.trace_plot(path=analysis_path)

        print("Generating posterior vs prior plots...")
        MCMC_test.posterior_vs_prior(path=analysis_path)

        print("Generating parameter pairplots...")
        MCMC_test.parameter_pairplot(path=analysis_path)
        
    print("Analysis complete.")


    
    