from pyscf.grad import rhf as rhf_grad
from pyscf.lib import logger
import numpy

class GradientsHF(rhf_grad.Gradients):

   def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        cput0 = (logger.process_clock(), logger.perf_counter())
        self.mol = self.base.mol
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst
        if atmlst is None:
           atmlst = range(self.mol.natm)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        # basically hcore_generator without get_hcore
        with_x2c = getattr(self.base, 'with_x2c', None)
        if with_x2c:
           Exception("I don't know what this is yet")
        else:
           with_ecp = self.mol.has_ecp()
           if with_ecp:
               ecp_atoms = set(self.mol._ecpbas[:,gto.ATOM_OF])
           else:
               ecp_atoms = ()
           aoslices = self.mol.aoslice_by_atom()
           dm0 = self.base.make_rdm1(mo_coeff, mo_occ)
           de = numpy.zeros((len(atmlst),3))
           for k, ia in enumerate(atmlst):
              p0, p1 = aoslices [ia,2:]
              with self.mol.with_rinv_at_nucleus(ia):
                  vrinv = self.mol.intor('int1e_iprinv', comp=3) # <\nabla|1/r|>
                  vrinv *= -self.mol.atom_charge(ia)
                  if with_ecp and ia in ecp_atoms:
                      vrinv += self.mol.intor('ECPscalar_iprinv', comp=3)
              vrinv = vrinv + vrinv.transpose(0,2,1)
              de[k] += numpy.einsum('xij,ij->x', vrinv, dm0)

        self.de = de + self.grad_nuc(atmlst=atmlst)
        if self.mol.symmetry:
            self.de = self.symmetrize(self.de, atmlst)
        logger.timer(self, 'SCF gradients', *cput0)
        self._finalize()
        return self.de
