import numpy as np
import pandas as pd
import copy

PATH = __file__[:-22]
PATH_TO_RAS_PARAMS = f'{PATH}/RAS_ODE_model_kinetic_parameters_v2.xlsx'

class KRAS:
    
    def __init__(self,name,multipliers=[1 for i in range(17)],color = 'grey'):
        
        self.name = name
        self.multipliers = multipliers
        self.color = color
        self.type = None

        self.volscale = 250
        self.kint = (3.5e-4)*multipliers[0]
        self.kdissD = (1.1e-4)*multipliers[1]
        self.kdissT = (2.5e-4)*multipliers[2]
        self.kassD = (2.3e6)*multipliers[3]
        
        self.kcat = (5.4)*multipliers[5]
        self.Km = (.23e-6)/self.volscale*multipliers[6]
        self.kD = 3.9*multipliers[7]
        self.KmD = (3.86e-4)/self.volscale*multipliers[8]
        self.KmT = (3e-4)/self.volscale*multipliers[9]
        self.Kd=(80e-9)*multipliers[11]
        self.kassEff=(4.5e7)*multipliers[12]
        
        #HaldaneGEF=self.kT*self.KmD/(self.kD*self.KmT)

        """
        if round(Haldaneint,ndigits=7) != round(HaldaneGEF,ndigits=7):
            print('Haldane coefficients not equal for {}'.format(self.name))
            print('Haldaneint: {}, HaldaneGEF: {}\n'.format(Haldaneint,HaldaneGEF))
        """
            
    def __getitem__(self,item:str):
        return self.__dict__[item]
    
    @property # ensures kdissEff is always a function of kassEff and Kd
    def kdissEff(self):
        return self.kassEff*self.Kd
    

class WT_(KRAS):

    def __init__(self,name,multipliers=[1 for i in range(17)],color = 'grey'):
        super().__init__(name,multipliers=multipliers,color=color)
        
        self.type = 'dep_kT'
        self.kassT = (2.2e6)*multipliers[4]

    @property # ensures kT is always a function of other params.
    def kT(self):
        Haldaneint=(self.kassD*self.kdissT)/(self.kdissD*self.kassT) # GTP, GDP not baked in.
        return self.kD*self.KmT*Haldaneint/self.KmD # GTP, GDP not baked in.

class Mutant_(KRAS):

    def __init__(self,name,kT,multipliers=[1 for i in range(17)],color = 'grey'):
        super().__init__(name,multipliers=multipliers,color=color)
        
        self.type = 'dep_kassT'
        self.kT = copy.deepcopy(kT)*multipliers[10]

    @property # ensures kassT is always a function of other params.
    def kassT(self):
        return self.kD*self.KmT*((self.kassD*self.kdissT)/(self.kdissD*self.kT))/self.KmD

WT = WT_('WT',color='forestgreen')
kT = WT.kT

try:
    mutants_df = pd.read_excel(PATH_TO_RAS_PARAMS,index_col=0).iloc[:, 0:-2]
except:
    print("Error loading mutant parameter excel file. Either manually set PATH_TO_RAS_PARAMS at top of RAS_ODE_models_MCMC or fix otherwise.")

def make_KRAS_Variant_from_index(index,color='grey',kT=None):
    return Mutant_(index,kT,list(mutants_df.loc[index])+[1]*3,color=color)

WT_Mut = Mutant_('WT_Mut',color='forestgreen',kT=kT)
A146T = make_KRAS_Variant_from_index('A146T',kT=kT) #passing in WT kT to specify that kassT is dependent.
A146V = make_KRAS_Variant_from_index('A146V',kT=kT)
A59T = make_KRAS_Variant_from_index('A59T',kT=kT)
F28L = make_KRAS_Variant_from_index('F28L',kT=kT)
G12A = make_KRAS_Variant_from_index('G12A',kT=kT)
G12C = make_KRAS_Variant_from_index('G12C',kT=kT,color='royalblue')
G12D = make_KRAS_Variant_from_index('G12D',kT=kT,color='cornflowerblue')
G12E = make_KRAS_Variant_from_index('G12E',kT=kT)
G12P = make_KRAS_Variant_from_index('G12P',kT=kT)
G12R = make_KRAS_Variant_from_index('G12R',kT=kT)
G12S = make_KRAS_Variant_from_index('G12S',kT=kT)
G12V = make_KRAS_Variant_from_index('G12V',kT=kT,color='lightskyblue')
G13C = make_KRAS_Variant_from_index('G13C',kT=kT)
G13D = make_KRAS_Variant_from_index('G13D',kT=kT,color='orchid')
G13S = make_KRAS_Variant_from_index('G13S',kT=kT)
G13V = make_KRAS_Variant_from_index('G13V',kT=kT)
Q61H = make_KRAS_Variant_from_index('Q61H',kT=kT)
Q61K = make_KRAS_Variant_from_index('Q61K',kT=kT)
Q61L = make_KRAS_Variant_from_index('Q61L',kT=kT,color='crimson')
Q61P = make_KRAS_Variant_from_index('Q61P',kT=kT)
Q61R = make_KRAS_Variant_from_index('Q61R',kT=kT)
Q61W = make_KRAS_Variant_from_index('Q61W',kT=kT)

all_mutants = [WT_Mut,A146T,A146V,A59T,F28L,G12A,G12C,G12D,G12E,G12P,G12R,G12S,G12V,G13C,G13D,G13S,G13V,Q61H,Q61K,Q61L,Q61P,Q61R,Q61W]  #A59T #10gly11 or 10dupG

def RAS_model(t,y,k):

    GD=y[0]
    GT=y[1]
    G0=y[2]
    Eff=y[3]
    GTEff=y[4]
    GDV=y[5]
    GTV=y[6]
    G0V=y[7]
    GTEffV=y[8]

    GAP = k[26]
    GDP = k[27]
    GTP = k[28]
    GEF = k[29]

    VmaxD=k[0]*GEF
    VmaxT=k[1]*GEF*GDP/GTP
    KmD=k[2]
    KmT=k[3]
    Vmax=k[4]*GAP
    Km=k[5]
    kint=k[6]
    kdissD=k[7]
    kdissT=k[8]
    kassDGDP=k[9]*GDP
    kassTGTP=k[10]*GTP
    kassEff=k[11]
    kdissEff=k[12]

    VmaxDV=k[13]*GEF
    VmaxTV=k[14]*GEF*GDP/GTP
    KmDV=k[15]
    KmTV=k[16]
    VmaxV=k[17]*GAP
    KmV=k[18]
    kintV=k[19]
    kdissDV=k[20]
    kdissTV=k[21]
    kassDGDPV=k[22]*GDP
    kassTGTPV=k[23]*GTP
    kassEffV=k[24]
    kdissEffV=k[25]

    R1=(VmaxD*GD/KmD-VmaxT*GT/KmT)/(1+GD/KmD+GT/KmT+GDV/KmDV+GTV/KmTV)
    R2=Vmax*GT/(Km*(1+GTV/KmV)+GT)
    R3=kint*GT
    R4=kdissD*GD-kassDGDP*G0
    R5=kdissT*GT-kassTGTP*G0
    R6=kassEff*GT*Eff-kdissEff*GTEff
    R7=kint*GTEff

    R8=(VmaxDV*GDV/KmDV-VmaxTV*GTV/KmTV)/(1+GD/KmD+GT/KmT+GDV/KmDV+GTV/KmTV)
    R9=VmaxV*GTV/(KmV*(1+GT/Km)+GTV)
    R10=kintV*GTV
    R11=kdissDV*GDV-kassDGDPV*G0V
    R12=kdissTV*GTV-kassTGTPV*G0V
    R13=kassEffV*GTV*Eff-kdissEffV*GTEffV
    R14=kintV*GTEffV

    dydt=[-R1+R2+R3-R4+R7,
        R1-R2-R3-R5-R6,
        R4+R5,
        -R6+R7-R13+R14,
        (R6-R7),
        (-R8+R9+R10-R11+R14),
        (R8-R9-R10-R12-R13),
        (R11+R12),
        (R13-R14)]
    return dydt

def get_params_RAS(get_param_opts):

    mutant = get_param_opts['mutant']
    fract_mut = get_param_opts['fract_mut']
    GTot = get_param_opts['GTot']
    EffTot = get_param_opts['EffTot']
    GAP = get_param_opts['GAP']
    GDP = get_param_opts['GDP']
    GTP = get_param_opts['GTP']
    GEF = get_param_opts['GEF']
    WT = WT_('WT')
    
    WTRasTot=(1-fract_mut)*GTot
    MutRasTot=fract_mut*GTot
    
    y0 = np.zeros(9)
    y0[0]=WTRasTot #GD=y(1);
    y0[1]=0 #GT=y(2);
    y0[2]=0 #G0=y(3);
    y0[3]=EffTot #Eff=y(4);
    y0[4]=0 #GTEff=y(5);
    y0[5]=MutRasTot #GDV=y(6);
    y0[6]=0 #GTV=y(7);
    y0[7]=0 #G0V=y(8);
    y0[8]=0 #GTVEff=y(9);

    k = np.ones(33)
    k[0] = WT.kD
    k[1] = WT.kT
    k[2] = WT.KmD
    k[3] = WT.KmT
    k[4] = WT.kcat
    k[5] = WT.Km
    k[6] = WT.kint
    k[7] = WT.kdissD
    k[8] = WT.kdissT
    k[9] = WT.kassD
    k[10] = WT.kassT
    k[11] = WT.kassEff
    k[12] = WT.kdissEff

    k[13] = mutant.kD
    k[14] = mutant.kT
    k[15] = mutant.KmD
    k[16] = mutant.KmT
    k[17] = mutant.kcat
    k[18] = mutant.Km
    k[19] = mutant.kint
    k[20] = mutant.kdissD
    k[21] = mutant.kdissT
    k[22] = mutant.kassD
    k[23] = mutant.kassT
    k[24] = mutant.kassEff
    k[25] = mutant.kdissEff

    k[26] = GAP
    k[27] = GDP
    k[28] = GTP
    k[29] = GEF

    k[30] = y0[0]+y0[1]+y0[2]+y0[4]
    k[31] = y0[5]+y0[6]+y0[7]+y0[8]
    k[32] = y0[3]+y0[4]+y0[8]

    return np.array(k),np.array(y0)

