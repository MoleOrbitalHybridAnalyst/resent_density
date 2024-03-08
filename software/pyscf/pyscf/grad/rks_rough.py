#import numpy
#from pyscf import gto
#from pyscf import lib
#from pyscf.lib import logger
from pyscf.grad import rhf_rough as rhfr_grad
from pyscf.grad import rks as rks_grad
#from pyscf import __config__

class GradientsHF(rks_grad.Gradients):
    kernel = rhfr_grad.GradientsHF.kernel


