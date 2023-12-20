import scipy.integrate as spi
import numpy as np
import math


STANDARD = lambda z : np.exp(-(z**2/2))/np.sqrt(2*np.pi)

######### EXPONENTIAL ############
l = 1
EXPONENTIAL = lambda z : l*np.exp(-l*z) if z >= 0 else 0