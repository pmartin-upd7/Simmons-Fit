# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:10:37 2024


        V (array) – Bias voltage

        J (float) – A/cm2

        phi (float) – barrier height in eV

        d (float) – barrier width in angstroms


@author: martin.pascal

"""
import matplotlib.pyplot as plt
import numpy as np
import lmfit
from lmfit import Model

def phi(x, phio, gamma):
    
    return (- gamma*1.602e-19*np.abs(x) + phio*1e-9)

def Simmons (x, d, phio, gamma):
    
    I = (1.602e-19/(2*3.14159*6.626e-34*d**2))*(
        (
            (phi(x, phio, gamma) - 1.602e-19*x/2)*
            np.exp(-(4*3.14159*d/6.626e-34)*
                   np.sqrt(2*9.109e-31*(phi(x, phio, gamma) - 1.602e-19*x/2))
                   )
            )-
        (
            (phi(x, phio, gamma) + 1.602e-19*x/2)*
            np.exp(-(4*3.14159*d/6.626e-34)*
                   np.sqrt(2*9.109e-31*(phi(x, phio, gamma) + 1.602e-19*x/2))
                   )
            )
        )

    return I

def residual(pars, x, original):
    model = Simmons(x, pars['d'], pars['phio'], pars['gamma'])
    return model - original


###############################################################################
# Lecture du fichier
data = np.loadtxt("Sample2-A1.txt", delimiter="\t")

# On prend la 3ᵉ colonne (index 2)
col = data[:, 2]
# Détecter les points où la valeur diminue (redémarrage)
change_points = np.where(np.diff(col) < 0)[0] + 1
# Découper
chunks = np.split(data, change_points)
chunk = chunks[2]  # chunks est ta liste
x = chunk[:,0]
y = chunk[:,1]/400e-12

###############################################################################
# add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
params = lmfit.Parameters()
params.add_many(('d', 15e-10, True, 5e-10, 50e-10, None, None),
                ('phio', 3e-10, True, 0.1e-10, 4e-10, None, None),
                ('gamma', 0.1, True, 0.05, 0.4, None, None)
    )
gmodel = Model(Simmons)
result = gmodel.fit(y, x=x, **params, nan_policy='omit')
y_eval = gmodel.eval(x=x, d=15e-10, phio=3e-10, gamma=0.1)


plt.plot(x,y, label='data')
plt.plot(x, y_eval, label='fit')
result.plot_fit()

print(result.fit_report())

plt.legend()   
