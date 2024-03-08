#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
import copy
import ctypes
import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf.gto import ATM_SLOTS, BAS_SLOTS, ATOM_OF, PTR_COORD
from pyscf.pbc.lib.kpts_helper import get_kconserv, get_kconserv3  # noqa
from pyscf import __config__

libpbc = lib.load_library('libpbc')
FFT_ENGINE = getattr(__config__, 'pbc_tools_pbc_fft_engine', 'FFTW')

def _fftn_blas(f, mesh):
    Gx = np.fft.fftfreq(mesh[0])
    Gy = np.fft.fftfreq(mesh[1])
    Gz = np.fft.fftfreq(mesh[2])
    expRGx = np.exp(np.einsum('x,k->xk', -2j*np.pi*np.arange(mesh[0]), Gx))
    expRGy = np.exp(np.einsum('x,k->xk', -2j*np.pi*np.arange(mesh[1]), Gy))
    expRGz = np.exp(np.einsum('x,k->xk', -2j*np.pi*np.arange(mesh[2]), Gz))
    out = np.empty(f.shape, dtype=np.complex128)
    #buf = np.empty(mesh, dtype=np.complex128)
    for i, fi in enumerate(f):
        #buf[:] = fi.reshape(mesh)
        buf = lib.copy(fi.reshape(mesh), dtype=np.complex128)
        g = lib.dot(buf.reshape(mesh[0],-1).T, expRGx, c=out[i].reshape(-1,mesh[0]))
        g = lib.dot(g.reshape(mesh[1],-1).T, expRGy, c=buf.reshape(-1,mesh[1]))
        g = lib.dot(g.reshape(mesh[2],-1).T, expRGz, c=out[i].reshape(-1,mesh[2]))
    return out.reshape(-1, *mesh)

def _ifftn_blas(g, mesh):
    Gx = np.fft.fftfreq(mesh[0])
    Gy = np.fft.fftfreq(mesh[1])
    Gz = np.fft.fftfreq(mesh[2])
    expRGx = np.exp(np.einsum('x,k->xk', 2j*np.pi*np.arange(mesh[0]), Gx))
    expRGy = np.exp(np.einsum('x,k->xk', 2j*np.pi*np.arange(mesh[1]), Gy))
    expRGz = np.exp(np.einsum('x,k->xk', 2j*np.pi*np.arange(mesh[2]), Gz))
    out = np.empty(g.shape, dtype=np.complex128)
    #buf = np.empty(mesh, dtype=np.complex128)
    for i, gi in enumerate(g):
        #buf[:] = gi.reshape(mesh)
        buf = lib.copy(gi.reshape(mesh), dtype=np.complex128)
        f = lib.dot(buf.reshape(mesh[0],-1).T, expRGx, 1./mesh[0], c=out[i].reshape(-1,mesh[0]))
        f = lib.dot(f.reshape(mesh[1],-1).T, expRGy, 1./mesh[1], c=buf.reshape(-1,mesh[1]))
        f = lib.dot(f.reshape(mesh[2],-1).T, expRGz, 1./mesh[2], c=out[i].reshape(-1,mesh[2]))
    return out.reshape(-1, *mesh)

if FFT_ENGINE == 'FFTW':
    try:
        libfft = lib.load_library('libfft')
    except OSError:
        raise RuntimeError("Failed to load libfft")

    def _complex_fftn_fftw(f, mesh, func):
        if f.dtype == np.double and f.flags.c_contiguous:
            f = lib.copy(f, dtype=np.complex128)
        else:
            f = np.asarray(f, order='C', dtype=np.complex128)
        mesh = np.asarray(mesh, order='C', dtype=np.int32)
        rank = len(mesh)
        out = np.empty_like(f)
        fn = getattr(libfft, func)
        for i, fi in enumerate(f):
            fn(fi.ctypes.data_as(ctypes.c_void_p),
               out[i].ctypes.data_as(ctypes.c_void_p),
               mesh.ctypes.data_as(ctypes.c_void_p),
               ctypes.c_int(rank))
        return out

    def _fftn_wrapper(a):
        mesh = a.shape[1:]
        return _complex_fftn_fftw(a, mesh, 'fft')
    def _ifftn_wrapper(a):
        mesh = a.shape[1:]
        return _complex_fftn_fftw(a, mesh, 'ifft')

elif FFT_ENGINE == 'PYFFTW':
    # pyfftw is slower than np.fft in most cases
    try:
        import pyfftw
        pyfftw.interfaces.cache.enable()
        nproc = lib.num_threads()
        def _fftn_wrapper(a):
            return pyfftw.interfaces.numpy_fft.fftn(a, axes=(1,2,3), threads=nproc)
        def _ifftn_wrapper(a):
            return pyfftw.interfaces.numpy_fft.ifftn(a, axes=(1,2,3), threads=nproc)
    except ImportError:
        def _fftn_wrapper(a):
            return np.fft.fftn(a, axes=(1,2,3))
        def _ifftn_wrapper(a):
            return np.fft.ifftn(a, axes=(1,2,3))

elif FFT_ENGINE == 'NUMPY':
    def _fftn_wrapper(a):
        return np.fft.fftn(a, axes=(1,2,3))
    def _ifftn_wrapper(a):
        return np.fft.ifftn(a, axes=(1,2,3))

elif FFT_ENGINE == 'NUMPY+BLAS':
    _EXCLUDE = [17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79,
                83, 89, 97,101,103,107,109,113,127,131,137,139,149,151,157,163,
                167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,
                257,263,269,271,277,281,283,293]
    _EXCLUDE = set(_EXCLUDE + [n*2 for n in _EXCLUDE] + [n*3 for n in _EXCLUDE])
    def _fftn_wrapper(a):
        mesh = a.shape[1:]
        if mesh[0] in _EXCLUDE and mesh[1] in _EXCLUDE and mesh[2] in _EXCLUDE:
            return _fftn_blas(a, mesh)
        else:
            return np.fft.fftn(a, axes=(1,2,3))
    def _ifftn_wrapper(a):
        mesh = a.shape[1:]
        if mesh[0] in _EXCLUDE and mesh[1] in _EXCLUDE and mesh[2] in _EXCLUDE:
            return _ifftn_blas(a, mesh)
        else:
            return np.fft.ifftn(a, axes=(1,2,3))

elif FFT_ENGINE == 'CUPY':
    import cupy
    def _fftn_wrapper(a):
        a = lib.device_put(a)
        a_fft = cupy.fft.fftn(a, axes=(1,2,3))
        a_fft = lib.device_get(a_fft, dtype=np.complex128)
        return a_fft
    def _ifftn_wrapper(a):
        a = lib.device_put(a)
        a_ifft = cupy.fft.ifftn(a, axes=(1,2,3))
        a_ifft = lib.device_get(a_ifft, dtype=np.complex128)
        return a_ifft

#?elif:  # 'FFTW+BLAS'
else:  # 'BLAS'
    def _fftn_wrapper(a):
        mesh = a.shape[1:]
        return _fftn_blas(a, mesh)
    def _ifftn_wrapper(a):
        mesh = a.shape[1:]
        return _ifftn_blas(a, mesh)


def fft(f, mesh):
    '''Perform the 3D FFT from real (R) to reciprocal (G) space.

    After FFT, (u, v, w) -> (j, k, l).
    (jkl) is in the index order of Gv.

    FFT normalization factor is 1., as in MH and in `numpy.fft`.

    Args:
        f : (nx*ny*nz,) ndarray
            The function to be FFT'd, flattened to a 1D array corresponding
            to the index order of :func:`cartesian_prod`.
        mesh : (3,) ndarray of ints (= nx,ny,nz)
            The number G-vectors along each direction.

    Returns:
        (nx*ny*nz,) ndarray
            The FFT 1D array in same index order as Gv (natural order of
            numpy.fft).

    '''
    if f.size == 0:
        return np.zeros_like(f)

    f3d = f.reshape(-1, *mesh)
    assert(f3d.shape[0] == 1 or f[0].size == f3d[0].size)
    g3d = _fftn_wrapper(f3d)
    ngrids = np.prod(mesh)
    if f.ndim == 1 or (f.ndim == 3 and f.size == ngrids):
        return g3d.ravel()
    else:
        return g3d.reshape(-1, ngrids)

def ifft(g, mesh):
    '''Perform the 3D inverse FFT from reciprocal (G) space to real (R) space.

    Inverse FFT normalization factor is 1./N, same as in `numpy.fft` but
    **different** from MH (they use 1.).

    Args:
        g : (nx*ny*nz,) ndarray
            The function to be inverse FFT'd, flattened to a 1D array
            corresponding to the index order of `span3`.
        mesh : (3,) ndarray of ints (= nx,ny,nz)
            The number G-vectors along each direction.

    Returns:
        (nx*ny*nz,) ndarray
            The inverse FFT 1D array in same index order as Gv (natural order
            of numpy.fft).

    '''
    if g.size == 0:
        return np.zeros_like(g)

    g3d = g.reshape(-1, *mesh)
    assert(g3d.shape[0] == 1 or g[0].size == g3d[0].size)
    f3d = _ifftn_wrapper(g3d)
    ngrids = np.prod(mesh)
    if g.ndim == 1 or (g.ndim == 3 and g.size == ngrids):
        return f3d.ravel()
    else:
        return f3d.reshape(-1, ngrids)


def fftk(f, mesh, expmikr):
    r'''Perform the 3D FFT of a real-space function which is (periodic*e^{ikr}).

    fk(k+G) = \sum_r fk(r) e^{-i(k+G)r} = \sum_r [f(k)e^{-ikr}] e^{-iGr}
    '''
    return fft(f*expmikr, mesh)


def ifftk(g, mesh, expikr):
    r'''Perform the 3D inverse FFT of f(k+G) into a function which is (periodic*e^{ikr}).

    fk(r) = (1/Ng) \sum_G fk(k+G) e^{i(k+G)r} = (1/Ng) \sum_G [fk(k+G)e^{iGr}] e^{ikr}
    '''
    return ifft(g, mesh) * expikr


def get_coulG(cell, k=np.zeros(3), exx=False, mf=None, mesh=None, Gv=None,
              wrap_around=True, omega=None, **kwargs):
    '''Calculate the Coulomb kernel for all G-vectors, handling G=0 and exchange.

    Args:
        k : (3,) ndarray
            k-point
        exx : bool or str
            Whether this is an exchange matrix element.
        mf : instance of :class:`SCF`

    Returns:
        coulG : (ngrids,) ndarray
            The Coulomb kernel.
        mesh : (3,) ndarray of ints (= nx,ny,nz)
            The number G-vectors along each direction.
        omega : float
            Enable Coulomb kernel erf(|omega|*r12)/r12 if omega > 0
            and erfc(|omega|*r12)/r12 if omega < 0.
            Note this parameter is slightly different to setting cell.omega
            for the treatment of exxdiv (at G0).  cell.omega affects Ewald
            probe charge at G0. It is used mostly with RSH functional for
            the long-range part of HF exchange. This parameter is used by
            real-space JK builder which requires Ewald probe charge to be
            computed with regular Coulomb interaction (1/r12) while the rest
            coulG is scaled as long-range Coulomb kernel.
    '''
    exxdiv = exx
    if isinstance(exx, str):
        exxdiv = exx
    elif exx and mf is not None:
        exxdiv = mf.exxdiv

    if mesh is None:
        mesh = cell.mesh
    if 'gs' in kwargs:
        warnings.warn('cell.gs is deprecated.  It is replaced by cell.mesh,'
                      'the number of PWs (=2*gs+1) along each direction.')
        mesh = [2*n+1 for n in kwargs['gs']]
    if Gv is None:
        Gv = cell.get_Gv(mesh)

    if abs(k).sum() > 1e-9:
        kG = k + Gv
    else:
        kG = Gv

    equal2boundary = None
    if wrap_around and abs(k).sum() > 1e-9:
        equal2boundary = np.zeros(Gv.shape[0], dtype=bool)
        # Here we 'wrap around' the high frequency k+G vectors into their lower
        # frequency counterparts.  Important if you want the gamma point and k-point
        # answers to agree
        b = cell.reciprocal_vectors()
        box_edge = np.einsum('i,ij->ij', np.asarray(mesh)//2+0.5, b)
        assert(all(np.linalg.solve(box_edge.T, k).round(9).astype(int)==0))
        reduced_coords = np.linalg.solve(box_edge.T, kG.T).T.round(9)
        on_edge = reduced_coords.astype(int)
        if cell.dimension >= 1:
            equal2boundary |= reduced_coords[:,0] == 1
            equal2boundary |= reduced_coords[:,0] ==-1
            kG[on_edge[:,0]== 1] -= 2 * box_edge[0]
            kG[on_edge[:,0]==-1] += 2 * box_edge[0]
        if cell.dimension >= 2:
            equal2boundary |= reduced_coords[:,1] == 1
            equal2boundary |= reduced_coords[:,1] ==-1
            kG[on_edge[:,1]== 1] -= 2 * box_edge[1]
            kG[on_edge[:,1]==-1] += 2 * box_edge[1]
        if cell.dimension == 3:
            equal2boundary |= reduced_coords[:,2] == 1
            equal2boundary |= reduced_coords[:,2] ==-1
            kG[on_edge[:,2]== 1] -= 2 * box_edge[2]
            kG[on_edge[:,2]==-1] += 2 * box_edge[2]

    #absG2 = np.einsum('gi,gi->g', kG, kG)
    absG2 = lib.multiply_sum(kG, kG, axis=1)

    if getattr(mf, 'kpts', None) is not None:
        kpts = mf.kpts
    else:
        kpts = k.reshape(1,3)
    Nk = len(kpts)

    if exxdiv == 'vcut_sph':  # PRB 77 193110
        Rc = (3*Nk*cell.vol/(4*np.pi))**(1./3)
        with np.errstate(divide='ignore',invalid='ignore'):
            coulG = 4*np.pi/absG2*(1.0 - np.cos(np.sqrt(absG2)*Rc))
        coulG[absG2==0] = 4*np.pi*0.5*Rc**2

        if cell.dimension < 3:
            raise NotImplementedError

    elif exxdiv == 'vcut_ws':  # PRB 87, 165122
        assert(cell.dimension == 3)
        if not getattr(mf, '_ws_exx', None):
            mf._ws_exx = precompute_exx(cell, kpts)
        exx_alpha = mf._ws_exx['alpha']
        exx_kcell = mf._ws_exx['kcell']
        exx_q = mf._ws_exx['q']
        exx_vq = mf._ws_exx['vq']

        with np.errstate(divide='ignore',invalid='ignore'):
            coulG = 4*np.pi/absG2*(1.0 - np.exp(-absG2/(4*exx_alpha**2)))
        coulG[absG2==0] = np.pi / exx_alpha**2
        # Index k+Gv into the precomputed vq and add on
        gxyz = np.dot(kG, exx_kcell.lattice_vectors().T)/(2*np.pi)
        gxyz = gxyz.round(decimals=6).astype(int)
        mesh = np.asarray(exx_kcell.mesh)
        gxyz = (gxyz + mesh)%mesh
        qidx = (gxyz[:,0]*mesh[1] + gxyz[:,1])*mesh[2] + gxyz[:,2]
        #qidx = [np.linalg.norm(exx_q-kGi,axis=1).argmin() for kGi in kG]
        maxqv = abs(exx_q).max(axis=0)
        is_lt_maxqv = (abs(kG) <= maxqv).all(axis=1)
        coulG = coulG.astype(exx_vq.dtype)
        coulG[is_lt_maxqv] += exx_vq[qidx[is_lt_maxqv]]

        if cell.dimension < 3:
            raise NotImplementedError

    else:
        # Ewald probe charge method to get the leading term of the finite size
        # error in exchange integrals

        G0_idx = np.where(absG2==0)[0]
        if cell.dimension != 2 or cell.low_dim_ft_type == 'inf_vacuum':
            with np.errstate(divide='ignore'):
                coulG = lib.multiply(4*np.pi, lib.reciprocal(absG2))
                coulG[G0_idx] = 0

        elif cell.dimension == 2:
            # The following 2D analytical fourier transform is taken from:
            # R. Sundararaman and T. Arias PRB 87, 2013
            b = cell.reciprocal_vectors()
            Ld2 = np.pi/np.linalg.norm(b[2])
            Gz = kG[:,2]
            Gp = np.linalg.norm(kG[:,:2], axis=1)
            weights = 1. - np.cos(Gz*Ld2) * np.exp(-Gp*Ld2)
            with np.errstate(divide='ignore', invalid='ignore'):
                coulG = weights*4*np.pi/absG2
            if len(G0_idx) > 0:
                coulG[G0_idx] = -2*np.pi*Ld2**2 #-pi*L_z^2/2

        elif cell.dimension == 1:
            logger.warn(cell, 'No method for PBC dimension 1, dim-type %s.'
                        '  cell.low_dim_ft_type="inf_vacuum"  should be set.',
                        cell.low_dim_ft_type)
            raise NotImplementedError

            # Carlo A. Rozzi, PRB 73, 205119 (2006)
            a = cell.lattice_vectors()
            # Rc is the cylindrical radius
            Rc = np.sqrt(cell.vol / np.linalg.norm(a[0])) / 2
            Gx = abs(kG[:,0])
            Gp = np.linalg.norm(kG[:,1:], axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                weights = 1 + Gp*Rc * scipy.special.j1(Gp*Rc) * scipy.special.k0(Gx*Rc)
                weights -= Gx*Rc * scipy.special.j0(Gp*Rc) * scipy.special.k1(Gx*Rc)
                coulG = 4*np.pi/absG2 * weights
                # TODO: numerical integation
                # coulG[Gx==0] = -4*np.pi * (dr * r * scipy.special.j0(Gp*r) * np.log(r)).sum()
            if len(G0_idx) > 0:
                coulG[G0_idx] = -np.pi*Rc**2 * (2*np.log(Rc) - 1)

        # The divergent part of periodic summation of (ii|ii) integrals in
        # Coulomb integrals were cancelled out by electron-nucleus
        # interaction. The periodic part of (ii|ii) in exchange cannot be
        # cancelled out by Coulomb integrals. Its leading term is calculated
        # using Ewald probe charge (the function madelung below)
        if cell.dimension > 0 and exxdiv == 'ewald' and len(G0_idx) > 0:
            coulG[G0_idx] += Nk*cell.vol*madelung(cell, kpts)

    if equal2boundary is not None:
        coulG[equal2boundary] = 0

    # Scale the coulG kernel for attenuated Coulomb integrals.
    # * omega is used by RealSpaceJKBuilder which requires ewald probe charge
    # being evaluated with regular Coulomb interaction (1/r12).
    # * cell.omega, which affects the ewald probe charge, is often set by
    # DFT-RSH functionals to build long-range HF-exchange for erf(omega*r12)/r12
    if omega is not None:
        if omega > 0:
            coulG *= np.exp(-.25/omega**2 * absG2)
        else:
            coulG *= (1 - np.exp(-.25/omega**2 * absG2))
    elif cell.omega != 0:
        coulG *= np.exp(-.25/cell.omega**2 * absG2)

    return coulG

def precompute_exx(cell, kpts):
    from pyscf.pbc import gto as pbcgto
    from pyscf.pbc.dft import gen_grid
    log = lib.logger.Logger(cell.stdout, cell.verbose)
    log.debug("# Precomputing Wigner-Seitz EXX kernel")
    Nk = get_monkhorst_pack_size(cell, kpts)
    log.debug("# Nk = %s", Nk)

    kcell = pbcgto.Cell()
    kcell.atom = 'H 0. 0. 0.'
    kcell.spin = 1
    kcell.unit = 'B'
    kcell.verbose = 0
    kcell.a = cell.lattice_vectors() * Nk
    Lc = 1.0/lib.norm(np.linalg.inv(kcell.a), axis=0)
    log.debug("# Lc = %s", Lc)
    Rin = Lc.min() / 2.0
    log.debug("# Rin = %s", Rin)
    # ASE:
    alpha = 5./Rin # sqrt(-ln eps) / Rc, eps ~ 10^{-11}
    log.info("WS alpha = %s", alpha)
    kcell.mesh = np.array([4*int(L*alpha*3.0) for L in Lc])  # ~ [120,120,120]
    # QE:
    #alpha = 3./Rin * np.sqrt(0.5)
    #kcell.mesh = (4*alpha*np.linalg.norm(kcell.a,axis=1)).astype(int)
    log.debug("# kcell.mesh FFT = %s", kcell.mesh)
    rs = gen_grid.gen_uniform_grids(kcell)
    kngs = len(rs)
    log.debug("# kcell kngs = %d", kngs)
    corners_coord = lib.cartesian_prod(([0, 1], [0, 1], [0, 1]))
    corners = np.dot(corners_coord, kcell.a)
    #vR = np.empty(kngs)
    #for i, rv in enumerate(rs):
    #    # Minimum image convention to corners of kcell parallelepiped
    #    r = lib.norm(rv-corners, axis=1).min()
    #    if np.isclose(r, 0.):
    #        vR[i] = 2*alpha / np.sqrt(np.pi)
    #    else:
    #        vR[i] = scipy.special.erf(alpha*r) / r
    r = np.min([lib.norm(rs-c, axis=1) for c in corners], axis=0)
    vR = scipy.special.erf(alpha*r) / (r+1e-200)
    vR[r<1e-9] = 2*alpha / np.sqrt(np.pi)
    vG = (kcell.vol/kngs) * fft(vR, kcell.mesh)

    if abs(vG.imag).max() > 1e-6:
        # vG should be real in regular lattice. If imaginary part is observed,
        # this probably means a ws cell was built from a unconventional
        # lattice. The SR potential erfc(alpha*r) for the charge in the center
        # of ws cell decays to the region out of ws cell. The Ewald-sum based
        # on the minimum image convention cannot be used to build the kernel
        # Eq (12) of PRB 87, 165122
        raise RuntimeError('Unconventional lattice was found')

    ws_exx = {'alpha': alpha,
              'kcell': kcell,
              'q'    : kcell.Gv,
              'vq'   : vG.real.copy()}
    log.debug("# Finished precomputing")
    return ws_exx


def madelung(cell, kpts):
    Nk = get_monkhorst_pack_size(cell, kpts)
    ecell = copy.copy(cell)
    ecell._atm = np.array([[1, cell._env.size, 0, 0, 0, 0]])
    ecell._env = np.append(cell._env, [0., 0., 0.])
    ecell.unit = 'B'
    #ecell.verbose = 0
    ecell.a = np.einsum('xi,x->xi', cell.lattice_vectors(), Nk)
    ecell.mesh = np.asarray(cell.mesh) * Nk

    if cell.omega == 0:
        ew_eta, ew_cut = ecell.get_ewald_params(cell.precision, ecell.mesh)
        lib.logger.debug1(cell, 'Monkhorst pack size %s ew_eta %s ew_cut %s',
                          Nk, ew_eta, ew_cut)
        return -2*ecell.ewald(ew_eta, ew_cut)

    else:
        # cell.ewald function does not use the Coulomb kernel function
        # get_coulG. When computing the nuclear interactions with attenuated
        # Coulomb operator, the Ewald summation technique is not needed
        # because the Coulomb kernel 4pi/G^2*exp(-G^2/4/omega**2) decays
        # quickly.
        Gv, Gvbase, weights = ecell.get_Gv_weights(ecell.mesh)
        coulG = get_coulG(ecell, Gv=Gv)
        ZSI = np.einsum("i,ij->j", ecell.atom_charges(), ecell.get_SI(Gv))
        return -np.einsum('i,i,i->', ZSI.conj(), ZSI, coulG*weights).real


def get_monkhorst_pack_size(cell, kpts):
    skpts = cell.get_scaled_kpts(kpts).round(decimals=6)
    Nk = np.array([len(np.unique(ki)) for ki in skpts.T])
    return Nk


def get_lattice_Ls(cell, nimgs=None, rcut=None, dimension=None, discard=True):
    '''Get the (Cartesian, unitful) lattice translation vectors for nearby images.
    The translation vectors can be used for the lattice summation.'''
    a = cell.lattice_vectors()
    b = cell.reciprocal_vectors(norm_to=1)
    heights_inv = lib.norm(b, axis=1)

    if nimgs is None:
        if rcut is None:
            rcut = cell.rcut
        # For atoms outside the cell, distance between certain basis of nearby
        # images may be smaller than rcut threshold even the corresponding Ls is
        # larger than rcut. The boundary penalty ensures that Ls would be able to
        # cover the basis that sitting out of the cell.
        # See issue https://github.com/pyscf/pyscf/issues/1017
        scaled_atom_coords = lib.dot(cell.atom_coords(), b.T)
        boundary_penalty = np.zeros((3,), dtype=float)
        neg = scaled_atom_coords.min(axis=0)
        neg_arg = neg < 0
        if neg_arg.any():
            boundary_penalty[neg_arg] = np.maximum(boundary_penalty[neg_arg], abs(neg[neg_arg]))
        pos = scaled_atom_coords.max(axis=0) - 1
        pos_arg = pos > 0
        if pos_arg.any():
            boundary_penalty[pos_arg] = np.maximum(boundary_penalty[pos_arg], pos[pos_arg])
        #boundary_penalty = np.max([abs(scaled_atom_coords).max(axis=0),
        #                           abs(1 - scaled_atom_coords).max(axis=0)], axis=0)
        nimgs = np.ceil(rcut * heights_inv + boundary_penalty).astype(int)
    else:
        rcut = max((np.asarray(nimgs))/heights_inv)

    if dimension is None:
        dimension = cell.dimension
    if dimension == 0:
        nimgs = [0, 0, 0]
    elif dimension == 1:
        nimgs = [nimgs[0], 0, 0]
    elif dimension == 2:
        nimgs = [nimgs[0], nimgs[1], 0]

    Ts = lib.cartesian_prod((np.arange(-nimgs[0], nimgs[0]+1),
                             np.arange(-nimgs[1], nimgs[1]+1),
                             np.arange(-nimgs[2], nimgs[2]+1)))
    Ls = np.dot(Ts, a)
    if discard:
        Ls = _discard_edge_images(cell, Ls, rcut)
    return np.asarray(Ls, order='C')

def _discard_edge_images(cell, Ls, rcut):
    '''
    Discard images if no basis in the image would contribute to lattice sum.
    '''
    if rcut <= 0:
        return np.zeros((1, 3))

    a = cell.lattice_vectors()
    scaled_atom_coords = np.linalg.solve(a.T, cell.atom_coords().T).T
    atom_boundary_max = scaled_atom_coords.max(axis=0)
    atom_boundary_min = scaled_atom_coords.min(axis=0)
    # ovlp_penalty ensures the overlap integrals for atoms in the adjcent
    # images are converged.
    ovlp_penalty = atom_boundary_max - atom_boundary_min
    # atom_boundary_min-1 ensures the values of basis at the grids on the edge
    # of the primitive cell converged
    boundary_max = np.ceil(np.max([atom_boundary_max  ,  ovlp_penalty], axis=0)).astype(int)
    boundary_min = np.floor(np.min([atom_boundary_min-1, -ovlp_penalty], axis=0)).astype(int)
    penalty_x = np.arange(boundary_min[0], boundary_max[0]+1)
    penalty_y = np.arange(boundary_min[1], boundary_max[1]+1)
    penalty_z = np.arange(boundary_min[2], boundary_max[2]+1)
    shifts = lib.cartesian_prod([penalty_x, penalty_y, penalty_z]).dot(a)
    Ls_mask = (np.linalg.norm(Ls + shifts[:,None,:], axis=2) < rcut).any(axis=0)
    # cell0 (Ls == 0) should always be included.
    Ls_mask[len(Ls)//2] = True
    return Ls[Ls_mask]


def super_cell(cell, ncopy):
    '''Create an ncopy[0] x ncopy[1] x ncopy[2] supercell of the input cell
    Note this function differs from :fun:`cell_plus_imgs` that cell_plus_imgs
    creates images in both +/- direction.

    Args:
        cell : instance of :class:`Cell`
        ncopy : (3,) array

    Returns:
        supcell : instance of :class:`Cell`
    '''
    a = cell.lattice_vectors()
    #:supcell.atom = []
    #:for Lx in range(ncopy[0]):
    #:    for Ly in range(ncopy[1]):
    #:        for Lz in range(ncopy[2]):
    #:            # Using cell._atom guarantees coord is in Bohr
    #:            for atom, coord in cell._atom:
    #:                L = np.dot([Lx, Ly, Lz], a)
    #:                supcell.atom.append([atom, coord + L])
    Ts = lib.cartesian_prod((np.arange(ncopy[0]),
                             np.arange(ncopy[1]),
                             np.arange(ncopy[2])))
    Ls = np.dot(Ts, a)
    supcell = cell.copy()
    supcell.a = np.einsum('i,ij->ij', ncopy, a)
    supcell.mesh = np.array([ncopy[0]*cell.mesh[0],
                             ncopy[1]*cell.mesh[1],
                             ncopy[2]*cell.mesh[2]])
    return _build_supcell_(supcell, cell, Ls)


def cell_plus_imgs(cell, nimgs):
    '''Create a supercell via nimgs[i] in each +/- direction, as in get_lattice_Ls().
    Note this function differs from :fun:`super_cell` that super_cell only
    stacks the images in + direction.

    Args:
        cell : instance of :class:`Cell`
        nimgs : (3,) array

    Returns:
        supcell : instance of :class:`Cell`
    '''
    a = cell.lattice_vectors()
    Ts = lib.cartesian_prod((np.arange(-nimgs[0], nimgs[0]+1),
                             np.arange(-nimgs[1], nimgs[1]+1),
                             np.arange(-nimgs[2], nimgs[2]+1)))
    Ls = np.dot(Ts, a)
    supcell = cell.copy()
    supcell.a = np.einsum('i,ij->ij', nimgs, a)
    supcell.mesh = np.array([(nimgs[0]*2+1)*cell.mesh[0],
                             (nimgs[1]*2+1)*cell.mesh[1],
                             (nimgs[2]*2+1)*cell.mesh[2]])
    return _build_supcell_(supcell, cell, Ls)

def _build_supcell_(supcell, cell, Ls):
    '''
    Construct supcell ._env directly without calling supcell.build() method.
    This reserves the basis contraction coefficients defined in cell
    '''
    nimgs = len(Ls)
    symbs = [atom[0] for atom in cell._atom] * nimgs
    coords = Ls.reshape(-1,1,3) + cell.atom_coords()
    supcell.atom = supcell._atom = list(zip(symbs, coords.reshape(-1,3).tolist()))
    supcell.unit = 'B'

    # Do not call supcell.build() since it may normalize the basis contraction
    # coefficients
    _env = np.append(cell._env, coords.ravel())
    _atm = np.repeat(cell._atm[None,:,:], nimgs, axis=0)
    _atm = _atm.reshape(-1, ATM_SLOTS)
    # Point to the corrdinates appended to _env
    _atm[:,PTR_COORD] = cell._env.size + np.arange(nimgs * cell.natm) * 3

    _bas = np.repeat(cell._bas[None,:,:], nimgs, axis=0)
    # For atom pointers in each image, shift natm*image_id
    _bas[:,:,ATOM_OF] += np.arange(nimgs)[:,None] * cell.natm

    supcell._atm = np.asarray(_atm, dtype=np.int32)
    supcell._bas = np.asarray(_bas.reshape(-1, BAS_SLOTS), dtype=np.int32)
    supcell._env = _env
    return supcell


def cutoff_to_mesh(a, cutoff):
    r'''
    Convert KE cutoff to FFT-mesh

        uses KE = k^2 / 2, where k_max ~ \pi / grid_spacing

    Args:
        a : (3,3) ndarray
            The real-space unit cell lattice vectors. Each row represents a
            lattice vector.
        cutoff : float
            KE energy cutoff in a.u.

    Returns:
        mesh : (3,) array
    '''
    b = 2 * np.pi * np.linalg.inv(a.T)
    cutoff = cutoff * _cubic2nonorth_factor(a)
    mesh = np.ceil(np.sqrt(2*cutoff)/lib.norm(b, axis=1) * 2).astype(int)
    return mesh

def mesh_to_cutoff(a, mesh):
    '''
    Convert #grid points to KE cutoff
    '''
    b = 2 * np.pi * np.linalg.inv(a.T)
    Gmax = lib.norm(b, axis=1) * np.asarray(mesh) * .5
    ke_cutoff = Gmax**2/2
    # scale down Gmax to get the real energy cutoff for non-orthogonal lattice
    return ke_cutoff / _cubic2nonorth_factor(a)

def _cubic2nonorth_factor(a):
    '''The factors to transform the energy cutoff from cubic lattice to
    non-orthogonal lattice. Energy cutoff is estimated based on cubic lattice.
    It needs to be rescaled for the non-orthogonal lattice to ensure that the
    minimal Gv vector in the reciprocal space is larger than the required
    energy cutoff.
    '''
    # Using ke_cutoff to set up a sphere, the sphere needs to be completely
    # inside the box defined by Gv vectors
    abase = a / np.linalg.norm(a, axis=1)[:,None]
    bbase = np.linalg.inv(abase.T)
    overlap = np.einsum('ix,ix->i', abase, bbase)
    return 1./overlap**2

def cutoff_to_gs(a, cutoff):
    '''Deprecated.  Replaced by function cutoff_to_mesh.'''
    return [n//2 for n in cutoff_to_mesh(a, cutoff)]

def gs_to_cutoff(a, gs):
    '''Deprecated.  Replaced by function mesh_to_cutoff.'''
    return mesh_to_cutoff(a, [2*n+1 for n in gs])

def gradient_gs(f_gs, Gv):
    '''
    Compute the G-space components of :math:`\nabla f(r)`
    given :math:`f(G)` and :math:`G`,
    which is equivalent to einsum('np,px->nxp', f_gs, 1j*Gv)
    '''
    ng, dim = Gv.shape
    if dim != 3:
        raise NotImplementedError
    Gv = np.asarray(Gv, order='C', dtype=float)
    f_gs = np.asarray(f_gs.reshape(-1,ng), order='C', dtype=np.complex128)
    n = f_gs.shape[0]
    out = np.empty((n,dim,ng), dtype=np.complex128)

    fn = getattr(libpbc, 'gradient_gs', None)
    try:
        fn(out.ctypes.data_as(ctypes.c_void_p),
           f_gs.ctypes.data_as(ctypes.c_void_p),
           Gv.ctypes.data_as(ctypes.c_void_p),
           ctypes.c_int(n), ctypes.c_size_t(ng))
    except Exception as e:
        raise RuntimeError("Failed to contract f_gs and iGv. %s" % e)
    return out

def gradient_by_fft(f, Gv, mesh):
    '''
    Compute :math:`\nabla f(r)` by FFT.
    '''
    ng, dim = Gv.shape
    assert ng == np.prod(mesh)
    f_gs = fft(f.reshape(-1,ng), mesh)
    f1_gs = gradient_gs(f_gs, Gv)
    f1_rs = ifft(f1_gs.reshape(-1,ng), mesh)
    f1_rs = lib.copy(f1_rs, dtype=float)
    f1 = f1_rs.reshape(-1,dim,ng)
    if f.ndim == 1:
        f1 = f1[0]
    return f1

def gradient_by_fdiff(cell, f, mesh=None):
    assert f.dtype == float
    if mesh is None:
        mesh = cell.mesh
    mesh = np.asarray(mesh, order='C', dtype=np.int32)
    a = cell.lattice_vectors()
    # cube
    dr = [a[i,i] / mesh[i] for i in range(3)]
    dr = np.asarray(dr, order='C', dtype=float)
    f = np.asarray(f, order='C')
    df = np.empty([3,*mesh], order='C', dtype=float)
    fun = getattr(libpbc, 'rs_gradient_cd3')
    fun(f.ctypes.data_as(ctypes.c_void_p),
        df.ctypes.data_as(ctypes.c_void_p),
        mesh.ctypes.data_as(ctypes.c_void_p),
        dr.ctypes.data_as(ctypes.c_void_p))
    return df.reshape(3,-1)

def laplacian_gs(f_gs, Gv):
    '''
    Compute the G-space components of :math:`\Delta f(r)`
    given :math:`f(G)` and :math:`G`,
    which is equivalent to einsum('np,p->np', f_gs, -G2)
    '''
    ng, dim = Gv.shape
    absG2 = lib.multiply_sum(Gv, Gv, axis=1)
    f_gs = f_gs.reshape(-1,ng)
    out = np.empty_like(f_gs, order='C', dtype=np.complex128)
    for i, f_gs_i in enumerate(f_gs):
        out[i] = lib.multiply(f_gs_i, -absG2, out=out[i])
    return out

def laplacian_by_fft(f, Gv, mesh):
    '''
    Compute :math:`\Delta f(r)` by FFT.
    '''
    ng, dim = Gv.shape
    assert ng == np.prod(mesh)
    f_gs = fft(f.reshape(-1,ng), mesh)
    f2_gs = laplacian_gs(f_gs, Gv)
    f2_rs = ifft(f2_gs.reshape(-1,ng), mesh)
    f2_rs = lib.copy(f2_rs, dtype=float)
    f2 = f2_rs.reshape(-1,ng)
    if f.ndim == 1:
        f2 = f2[0]
    return f2

def laplacian_by_fdiff(cell, f, mesh=None):
    assert f.dtype == float
    if mesh is None:
        mesh = cell.mesh
    mesh = np.asarray(mesh, order='C', dtype=np.int32)
    assert f.size == np.prod(mesh)
    a = cell.lattice_vectors()
    # cube
    dr = [a[i,i] / mesh[i] for i in range(3)]
    dr = np.asarray(dr, order='C', dtype=float)
    f = np.asarray(f, order='C')
    lf = np.empty_like(f)
    fun = getattr(libpbc, 'rs_laplacian_cd3')
    fun(f.ctypes.data_as(ctypes.c_void_p),
        lf.ctypes.data_as(ctypes.c_void_p),
        mesh.ctypes.data_as(ctypes.c_void_p),
        dr.ctypes.data_as(ctypes.c_void_p))
    return lf.ravel()

def solve_poisson(cell, rho, coulG=None, Gv=None, mesh=None,
                  compute_potential=True, compute_gradient=False,
                  real_potential=True):
    if mesh is None:
        mesh = cell.mesh
    if coulG is None:
        coulG = get_coulG(cell, mesh=mesh)

    ng = np.prod(mesh)
    rhoG = fft(rho.reshape(-1,ng), mesh)
    phiR = None
    if compute_potential:
        phiG = np.empty_like(rhoG)
        for i in range(len(phiG)):
            phiG[i] = lib.multiply(rhoG[i], coulG)
        phiR = ifft(phiG, mesh)
        if real_potential:
            phiR = lib.copy(phiR, dtype=float)

    dphiR = None
    if compute_gradient:
        if Gv is None:
            Gv = cell.get_Gv(mesh)
        _ng, dim = Gv.shape
        assert _ng == ng
        if dim != 3:
            raise NotImplementedError

        drhoG = gradient_gs(rhoG, Gv).reshape(-1,dim,ng)
        dphiR = np.empty_like(drhoG, dtype=np.complex128)
        for i in range(len(dphiR)):
            for x in range(dim):
                dphiG = lib.multiply(drhoG[i,x], coulG)
                dphiR[i,x] = ifft(dphiG, mesh)
        if real_potential:
            dphiR = lib.copy(dphiR, dtype=float)

    if rho.ndim == 1:
        if compute_potential:
            phiR = phiR[0]
        if compute_gradient:
            dphiR = dphiR[0]
    return phiR, dphiR

def restrict_by_fft(f, mesh, submeshes):
    from pyscf.pbc.dft.multigrid.utils import _take_4d
    real = False
    if f.dtype == float:
        real = True

    ng = np.prod(mesh)
    f_gs = fft(f.reshape(-1,ng), mesh).reshape(-1,*mesh)

    out = []
    submeshes = np.asarray(submeshes).reshape(-1,3)
    for i, submesh in enumerate(submeshes):
        ng_sub = np.prod(submesh)
        gx = np.fft.fftfreq(submesh[0], 1./submesh[0]).astype(np.int32)
        gy = np.fft.fftfreq(submesh[1], 1./submesh[1]).astype(np.int32)
        gz = np.fft.fftfreq(submesh[2], 1./submesh[2]).astype(np.int32)
        f_sub_gs = _take_4d(f_gs, (None, gx, gy, gz)).reshape(-1,ng_sub)
        f_sub_rs = ifft(f_sub_gs, submesh)
        if real:
            f_sub_rs = lib.copy(f_sub_rs, dtype=float)
        fac = ng_sub / ng
        f_sub_rs = lib.multiply(fac, f_sub_rs, out=f_sub_rs)
        if f.ndim == 1:
            f_sub_rs = f_sub_rs[0]
        out.append(f_sub_rs)

    if len(submeshes) == 1:
        out = out[0]
    return out

def prolong_by_fft(f_sub, mesh, submesh):
    from pyscf.pbc.dft.multigrid.utils import _takebak_4d
    real = False
    if f_sub.dtype == float:
        real = True
    input_is_vector = f_sub.ndim == 1

    ng = np.prod(mesh)
    ng_sub = np.prod(submesh)
    f_sub = f_sub.reshape(-1,ng_sub)
    nset = f_sub.shape[0]
    f_sub_gs = fft(f_sub, submesh)
    gx = np.fft.fftfreq(submesh[0], 1./submesh[0]).astype(np.int32)
    gy = np.fft.fftfreq(submesh[1], 1./submesh[1]).astype(np.int32)
    gz = np.fft.fftfreq(submesh[2], 1./submesh[2]).astype(np.int32)
    f_gs = np.zeros([nset, *mesh], dtype=np.complex128)
    f_gs = _takebak_4d(f_gs, f_sub_gs.reshape(-1,*submesh), (None, gx, gy, gz)).reshape(-1,ng)
    f_rs = ifft(f_gs, mesh)
    if real:
        f_rs = lib.copy(f_rs, dtype=float)
    fac = ng / ng_sub
    f_rs = lib.multiply(fac, f_rs, out=f_rs)

    if input_is_vector:
        f_rs = f_rs[0]
    return f_rs
