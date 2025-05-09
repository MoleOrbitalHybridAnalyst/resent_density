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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Extension to numpy and scipy
'''

import string
import ctypes
import math
import numpy
from pyscf.lib import misc
from numpy import asarray  # For backward compatibility

EINSUM_MAX_SIZE = getattr(misc.__config__, 'lib_einsum_max_size', 2000)

try:
    # Import tblis before libnp_helper to avoid potential dl-loading conflicts
    from pyscf import tblis_einsum
    FOUND_TBLIS = True
except (ImportError, OSError):
    FOUND_TBLIS = False

CUBLAS = getattr(misc.__config__, 'lib_cublas', False)
if CUBLAS:
    try:
        # use cupy as the backend for now
        import cupy
    except ImportError:
        raise ImportError("CUBLAS backend is not available.")


_np_helper = misc.load_library('libnp_helper')

BLOCK_DIM = 192
PLAIN = 0
HERMITIAN = 1
ANTIHERMI = 2
SYMMETRIC = 3

LeviCivita = numpy.zeros((3,3,3))
LeviCivita[0,1,2] = LeviCivita[1,2,0] = LeviCivita[2,0,1] = 1
LeviCivita[0,2,1] = LeviCivita[2,1,0] = LeviCivita[1,0,2] = -1

PauliMatrices = numpy.array([[[0., 1.],
                              [1., 0.]],  # x
                             [[0.,-1j],
                              [1j, 0.]],  # y
                             [[1., 0.],
                              [0.,-1.]]]) # z

if hasattr(numpy, 'einsum_path'):
    _einsum_path = numpy.einsum_path
else:
    def _einsum_path(subscripts, *operands, **kwargs):
        #indices  = re.split(',|->', subscripts)
        #indices_in = indices[:-1]
        #idx_final = indices[-1]
        if '->' in subscripts:
            indices_in, idx_final = subscripts.split('->')
            indices_in = indices_in.split(',')
            # indices = indices_in + [idx_final]
        else:
            idx_final = ''
            indices_in = subscripts.split('->')[0].split(',')

        if len(indices_in) <= 2:
            idx_removed = set(indices_in[0]).intersection(set(indices_in[1]))
            einsum_str = indices_in[1] + ',' + indices_in[0] + '->' + idx_final
            return operands, [((1,0), idx_removed, einsum_str, idx_final)]

        input_sets = [set(x) for x in indices_in]
        n_shared_max = 0
        for i in range(len(indices_in)):
            for j in range(i):
                tmp = input_sets[i].intersection(input_sets[j])
                n_shared_indices = len(tmp)
                if n_shared_indices > n_shared_max:
                    n_shared_max = n_shared_indices
                    idx_removed = tmp
                    a,b = i,j

        idxA = indices_in.pop(a)
        idxB = indices_in.pop(b)
        rest_idx = ''.join(indices_in) + idx_final
        idx_out = input_sets[a].union(input_sets[b])
        idx_out = ''.join(idx_out.intersection(set(rest_idx)))

        indices_in.append(idx_out)
        einsum_str = idxA + ',' + idxB + '->' + idx_out
        einsum_args = _einsum_path(','.join(indices_in)+'->'+idx_final)[1]
        einsum_args.insert(0, ((a, b), idx_removed, einsum_str, indices_in))
        return operands, einsum_args

_numpy_einsum = numpy.einsum
def _contract(subscripts, *tensors, **kwargs):
    idx_str = subscripts.replace(' ','')
    A, B = tensors
    # Call numpy.asarray because A or B may be HDF5 Datasets
    A = numpy.asarray(A)
    B = numpy.asarray(B)

    # small problem size
    if A.size < EINSUM_MAX_SIZE or B.size < EINSUM_MAX_SIZE:
        return _numpy_einsum(idx_str, A, B)

    C_dtype = numpy.result_type(A, B)
    if FOUND_TBLIS and C_dtype == numpy.double:
        # tblis is slow for complex type
        return tblis_einsum.contract(idx_str, A, B, **kwargs)

    indices  = idx_str.replace(',', '').replace('->', '')
    if '->' not in idx_str or any(indices.count(x) != 2 for x in set(indices)):
        return _numpy_einsum(idx_str, A, B)

    # Split the strings into a list of idx char's
    idxA, idxBC = idx_str.split(',')
    idxB, idxC = idxBC.split('->')
    assert len(idxA) == A.ndim
    assert len(idxB) == B.ndim

    uniq_idxA = set(idxA)
    uniq_idxB = set(idxB)
    # Find the shared indices being summed over
    shared_idxAB = uniq_idxA.intersection(uniq_idxB)

    if ((not shared_idxAB) or  # Indices must overlap
        # one operand is a subset of the other one (e.g. 'ijkl,jk->il')
        uniq_idxA == shared_idxAB or uniq_idxB == shared_idxAB or
        # repeated indices (e.g. 'iijk,kl->jl')
        len(idxA) != len(uniq_idxA) or len(idxB) != len(uniq_idxB)):
        return _numpy_einsum(idx_str, A, B)

    DEBUG = kwargs.get('DEBUG', False)

    if DEBUG:
        print("*** Einsum for", idx_str)
        print(" idxA =", idxA)
        print(" idxB =", idxB)
        print(" idxC =", idxC)

    # Get the range for each index and put it in a dictionary
    rangeA = dict(zip(idxA, A.shape))
    rangeB = dict(zip(idxB, B.shape))
    #rangeC = dict(zip(idxC, C.shape))
    if DEBUG:
        print("rangeA =", rangeA)
        print("rangeB =", rangeB)

    idxAt = list(idxA)
    idxBt = list(idxB)
    inner_shape = 1
    insert_B_loc = 0
    for n in shared_idxAB:
        if rangeA[n] != rangeB[n]:
            err = ('ERROR: In index string %s, the range of index %s is '
                   'different in A (%d) and B (%d)' %
                   (idx_str, n, rangeA[n], rangeB[n]))
            raise ValueError(err)

        # Bring idx all the way to the right for A
        # and to the left (but preserve order) for B
        idxA_n = idxAt.index(n)
        idxAt.insert(len(idxAt)-1, idxAt.pop(idxA_n))

        idxB_n = idxBt.index(n)
        idxBt.insert(insert_B_loc, idxBt.pop(idxB_n))
        insert_B_loc += 1

        inner_shape *= rangeA[n]

    if DEBUG:
        print("shared_idxAB =", shared_idxAB)
        print("inner_shape =", inner_shape)

    # Transpose the tensors into the proper order and reshape into matrices
    new_orderA = [idxA.index(idx) for idx in idxAt]
    new_orderB = [idxB.index(idx) for idx in idxBt]

    if DEBUG:
        print("Transposing A as", new_orderA)
        print("Transposing B as", new_orderB)
        print("Reshaping A as (-1,", inner_shape, ")")
        print("Reshaping B as (", inner_shape, ",-1)")

    shapeCt = list()
    idxCt = list()
    for idx in idxAt:
        if idx in shared_idxAB:
            break
        shapeCt.append(rangeA[idx])
        idxCt.append(idx)
    for idx in idxBt:
        if idx in shared_idxAB:
            continue
        shapeCt.append(rangeB[idx])
        idxCt.append(idx)
    new_orderCt = [idxCt.index(idx) for idx in idxC]

    if A.size == 0 or B.size == 0:
        shapeCt = [shapeCt[i] for i in new_orderCt]
        return numpy.zeros(shapeCt, dtype=C_dtype)

    At = A.transpose(new_orderA)
    Bt = B.transpose(new_orderB)

    if At.flags.f_contiguous:
        At = numpy.asarray(At.reshape(-1,inner_shape), order='F')
    else:
        At = numpy.asarray(At.reshape(-1,inner_shape), order='C')
    if Bt.flags.f_contiguous:
        Bt = numpy.asarray(Bt.reshape(inner_shape,-1), order='F')
    else:
        Bt = numpy.asarray(Bt.reshape(inner_shape,-1), order='C')

    return dot(At,Bt).reshape(shapeCt, order='A').transpose(new_orderCt)

def einsum(subscripts, *tensors, **kwargs):
    '''Perform a more efficient einsum via reshaping to a matrix multiply.

    Current differences compared to numpy.einsum:
    This assumes that each repeated index is actually summed (i.e. no 'i,i->i')
    and appears only twice (i.e. no 'ij,ik,il->jkl'). The output indices must
    be explicitly specified (i.e. 'ij,j->i' and not 'ij,j').
    '''
    contract = kwargs.pop('_contract', _contract)

    subscripts = subscripts.replace(' ','')
    if len(tensors) <= 1 or '...' in subscripts:
        out = _numpy_einsum(subscripts, *tensors, **kwargs)
    elif len(tensors) <= 2:
        out = _contract(subscripts, *tensors, **kwargs)
    else:
        if '->' in subscripts:
            indices_in, idx_final = subscripts.split('->')
            indices_in = indices_in.split(',')
        else:
            # idx_final = ''
            indices_in = subscripts.split('->')[0].split(',')
        tensors = list(tensors)
        contraction_list = _einsum_path(subscripts, *tensors, optimize=True,
                                        einsum_call=True)[1]
        for contraction in contraction_list:
            inds, idx_rm, einsum_str, remaining = contraction[:4]
            tmp_operands = [tensors.pop(x) for x in inds]
            if len(tmp_operands) > 2:
                out = _numpy_einsum(einsum_str, *tmp_operands)
            else:
                out = contract(einsum_str, *tmp_operands)
            tensors.append(out)
    return out


# 2d -> 1d or 3d -> 2d
def pack_tril(mat, axis=-1, out=None):
    '''flatten the lower triangular part of a matrix.
    Given mat, it returns mat[...,numpy.tril_indices(mat.shape[0])]

    Examples:

    >>> pack_tril(numpy.arange(9).reshape(3,3))
    [0 3 4 6 7 8]
    '''
    if mat.size == 0:
        return numpy.zeros(mat.shape+(0,), dtype=mat.dtype)

    if mat.ndim == 2:
        count, nd = 1, mat.shape[0]
        shape = nd*(nd+1)//2
    else:
        count, nd = mat.shape[:2]
        shape = (count, nd*(nd+1)//2)

    if mat.ndim == 2 or axis == -1:
        mat = numpy.asarray(mat, order='C')
        out = numpy.ndarray(shape, mat.dtype, buffer=out)
        if mat.dtype == numpy.double:
            fn = _np_helper.NPdpack_tril_2d
        elif mat.dtype == numpy.complex128:
            fn = _np_helper.NPzpack_tril_2d
        else:
            out[:] = mat[numpy.tril_indices(nd)]
            return out

        fn(ctypes.c_int(count), ctypes.c_int(nd),
           out.ctypes.data_as(ctypes.c_void_p),
           mat.ctypes.data_as(ctypes.c_void_p))
        return out

    else:  # pack the leading two dimension
        assert(axis == 0)
        out = mat[numpy.tril_indices(nd)]
        return out

# 1d -> 2d or 2d -> 3d, write hermitian lower triangle to upper triangle
def unpack_tril(tril, filltriu=HERMITIAN, axis=-1, out=None):
    '''Reversed operation of pack_tril.

    Kwargs:
        filltriu : int

            | 0           Do not fill the upper triangular part, random number may appear
                          in the upper triangular part
            | 1 (default) Transpose the lower triangular part to fill the upper triangular part
            | 2           Similar to filltriu=1, negative of the lower triangular part is assign
                          to the upper triangular part to make the matrix anti-hermitian

    Examples:

    >>> unpack_tril(numpy.arange(6.))
    [[ 0. 1. 3.]
     [ 1. 2. 4.]
     [ 3. 4. 5.]]
    >>> unpack_tril(numpy.arange(6.), 0)
    [[ 0. 0. 0.]
     [ 1. 2. 0.]
     [ 3. 4. 5.]]
    >>> unpack_tril(numpy.arange(6.), 2)
    [[ 0. -1. -3.]
     [ 1.  2. -4.]
     [ 3.  4.  5.]]
    '''
    tril = numpy.asarray(tril, order='C')
    if tril.ndim == 1:
        count, nd = 1, tril.size
        nd = int(numpy.sqrt(nd*2))
        shape = (nd,nd)
    elif tril.ndim == 2:
        if axis == 0:
            nd, count = tril.shape
        else:
            count, nd = tril.shape
        nd = int(numpy.sqrt(nd*2))
        shape = (count,nd,nd)
    else:
        raise NotImplementedError('unpack_tril for high dimension arrays')

    if (tril.dtype != numpy.double and tril.dtype != numpy.complex128):
        out = numpy.ndarray(shape, tril.dtype, buffer=out)
        idx, idy = numpy.tril_indices(nd)
        if filltriu == ANTIHERMI:
            out[...,idy,idx] = -tril
        else:
            out[...,idy,idx] = tril
        out[...,idx,idy] = tril
        return out

    elif tril.ndim == 1 or axis == -1 or axis == tril.ndim-1:
        out = numpy.ndarray(shape, tril.dtype, buffer=out)
        if tril.dtype == numpy.double:
            fn = _np_helper.NPdunpack_tril_2d
        else:
            fn = _np_helper.NPzunpack_tril_2d
        fn(ctypes.c_int(count), ctypes.c_int(nd),
           tril.ctypes.data_as(ctypes.c_void_p),
           out.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(filltriu))
        return out

    else:  # unpack the leading dimension
        assert(axis == 0)
        shape = (nd,nd) + tril.shape[1:]
        out = numpy.ndarray(shape, tril.dtype, buffer=out)
        idx = numpy.tril_indices(nd)
        if filltriu == HERMITIAN:
            for ij,(i,j) in enumerate(zip(*idx)):
                out[i,j] = tril[ij]
                out[j,i] = tril[ij].conj()
        elif filltriu == ANTIHERMI:
            for ij,(i,j) in enumerate(zip(*idx)):
                out[i,j] = tril[ij]
                out[j,i] =-tril[ij].conj()
        elif filltriu == SYMMETRIC:
            #:for ij,(i,j) in enumerate(zip(*idx)):
            #:    out[i,j] = out[j,i] = tril[ij]
            idxy = numpy.empty((nd,nd), dtype=numpy.int)
            idxy[idx[0],idx[1]] = idxy[idx[1],idx[0]] = numpy.arange(nd*(nd+1)//2)
            numpy.take(tril, idxy, axis=0, out=out)
        else:
            out[idx] = tril
        return out

# extract a row from a tril-packed matrix
def unpack_row(tril, row_id):
    '''Extract one row of the lower triangular part of a matrix.
    It is equivalent to unpack_tril(a)[row_id]

    Examples:

    >>> unpack_row(numpy.arange(6.), 0)
    [ 0. 1. 3.]
    >>> unpack_tril(numpy.arange(6.))[0]
    [ 0. 1. 3.]
    '''
    tril = numpy.ascontiguousarray(tril)
    nd = int(numpy.sqrt(tril.size*2))
    mat = numpy.empty(nd, tril.dtype)
    if tril.dtype == numpy.double:
        fn = _np_helper.NPdunpack_row
    elif tril.dtype == numpy.complex128:
        fn = _np_helper.NPzunpack_row
    else:
        p0 = row_id*(row_id+1)//2
        p1 = row_id*(row_id+1)//2 + row_id
        idx = numpy.arange(row_id, nd)
        return numpy.append(tril[p0:p1], tril[idx*(idx+1)//2+row_id])

    fn.restype = ctypes.c_void_p
    fn(ctypes.c_int(nd), ctypes.c_int(row_id),
       tril.ctypes.data_as(ctypes.c_void_p),
       mat.ctypes.data_as(ctypes.c_void_p))
    return mat

# for i > j of 2d mat, mat[j,i] = mat[i,j]
def hermi_triu(mat, hermi=HERMITIAN, inplace=True):
    '''Use the elements of the lower triangular part to fill the upper triangular part.

    Kwargs:
        filltriu : int

            | 1 (default) return a hermitian matrix
            | 2           return an anti-hermitian matrix

    Examples:

    >>> unpack_row(numpy.arange(9.).reshape(3,3), 1)
    [[ 0.  3.  6.]
     [ 3.  4.  7.]
     [ 6.  7.  8.]]
    >>> unpack_row(numpy.arange(9.).reshape(3,3), 2)
    [[ 0. -3. -6.]
     [ 3.  4. -7.]
     [ 6.  7.  8.]]
    '''
    assert(hermi == HERMITIAN or hermi == ANTIHERMI)
    if not inplace:
        mat = mat.copy('A')
    if mat.flags.c_contiguous:
        buf = mat
    elif mat.flags.f_contiguous:
        buf = mat.T
    else:
        raise NotImplementedError

    nd = mat.shape[0]
    assert(mat.size == nd**2)

    if mat.dtype == numpy.double:
        fn = _np_helper.NPdsymm_triu
    elif mat.dtype == numpy.complex128:
        fn = _np_helper.NPzhermi_triu
    else:
        raise NotImplementedError
    fn.restype = ctypes.c_void_p
    fn(ctypes.c_int(nd), buf.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(hermi))
    return mat


LINEAR_DEP_THRESHOLD = 1e-10
def solve_lineq_by_SVD(a, b):
    '''Solving a * x = b.  If a is a singular matrix, its small SVD values are
    neglected.
    '''
    t, w, vH = numpy.linalg.svd(a)
    idx = []
    for i,wi in enumerate(w):
        if wi > LINEAR_DEP_THRESHOLD:
            idx.append(i)
    if idx:
        idx = numpy.array(idx)
        tb = numpy.dot(numpy.array(t[:,idx]).T.conj(), numpy.array(b))
        x = numpy.dot(numpy.array(vH[idx,:]).T.conj(), tb / w[idx])
    else:
        x = numpy.zeros_like(b)
    return x

def take_2d(a, idx, idy, out=None):
    '''Equivalent to a[idx[:,None],idy] for a 2D array.

    Examples:

    >>> out = numpy.arange(9.).reshape(3,3)
    >>> take_2d(a, [0,2], [0,2])
    [[ 0.  2.]
     [ 6.  8.]]
    '''
    a = numpy.asarray(a, order='C')
    out = numpy.ndarray((len(idx),len(idy)), dtype=a.dtype, buffer=out)
    idx = numpy.asarray(idx, dtype=numpy.int32)
    idy = numpy.asarray(idy, dtype=numpy.int32)
    if a.dtype == numpy.double:
        fn = _np_helper.NPdtake_2d
    elif a.dtype == numpy.complex128:
        fn = _np_helper.NPztake_2d
    else:
        return a[idx[:,None],idy]
    fn(out.ctypes.data_as(ctypes.c_void_p),
       a.ctypes.data_as(ctypes.c_void_p),
       idx.ctypes.data_as(ctypes.c_void_p),
       idy.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(out.shape[1]), ctypes.c_int(a.shape[1]),
       ctypes.c_int(idx.size), ctypes.c_int(idy.size))
    return out

def takebak_2d(out, a, idx, idy, thread_safe=True):
    '''Reverse operation of take_2d.  Equivalent to out[idx[:,None],idy] += a
    for a 2D array.

    Examples:

    >>> out = numpy.zeros((3,3))
    >>> takebak_2d(out, numpy.ones((2,2)), [0,2], [0,2])
    [[ 1.  0.  1.]
     [ 0.  0.  0.]
     [ 1.  0.  1.]]
    '''
    assert(out.flags.c_contiguous)
    a = numpy.asarray(a, order='C')
    if out.dtype != a.dtype:
        a = a.astype(out.dtype)
    if out.dtype == numpy.double:
        fn = _np_helper.NPdtakebak_2d
    elif out.dtype == numpy.complex128:
        fn = _np_helper.NPztakebak_2d
    else:
        if thread_safe:
            out[idx[:,None], idy] += a
        else:
            raise NotImplementedError
        return out
    idx = numpy.asarray(idx, dtype=numpy.int32)
    idy = numpy.asarray(idy, dtype=numpy.int32)
    fn(out.ctypes.data_as(ctypes.c_void_p),
       a.ctypes.data_as(ctypes.c_void_p),
       idx.ctypes.data_as(ctypes.c_void_p),
       idy.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(out.shape[1]), ctypes.c_int(a.shape[1]),
       ctypes.c_int(idx.size), ctypes.c_int(idy.size),
       ctypes.c_int(thread_safe))
    return out

def transpose(a, axes=None, inplace=False, out=None):
    '''Transposing an array with better memory efficiency

    Examples:

    >>> transpose(numpy.ones((3,2)))
    [[ 1.  1.  1.]
     [ 1.  1.  1.]]
    '''
    if inplace:
        arow, acol = a.shape
        assert(arow == acol)
        tmp = numpy.empty((BLOCK_DIM,BLOCK_DIM))
        for c0, c1 in misc.prange(0, acol, BLOCK_DIM):
            for r0, r1 in misc.prange(0, c0, BLOCK_DIM):
                tmp[:c1-c0,:r1-r0] = a[c0:c1,r0:r1]
                a[c0:c1,r0:r1] = a[r0:r1,c0:c1].T
                a[r0:r1,c0:c1] = tmp[:c1-c0,:r1-r0].T
            # diagonal blocks
            a[c0:c1,c0:c1] = a[c0:c1,c0:c1].T
        return a

    if (not a.flags.c_contiguous
        or (a.dtype != numpy.double and a.dtype != numpy.complex128)):
        if a.ndim == 2:
            arow, acol = a.shape
            out = numpy.empty((acol,arow), a.dtype)
            r1 = c1 = 0
            for c0 in range(0, acol-BLOCK_DIM, BLOCK_DIM):
                c1 = c0 + BLOCK_DIM
                for r0 in range(0, arow-BLOCK_DIM, BLOCK_DIM):
                    r1 = r0 + BLOCK_DIM
                    out[c0:c1,r0:r1] = a[r0:r1,c0:c1].T
                out[c0:c1,r1:arow] = a[r1:arow,c0:c1].T
            for r0 in range(0, arow-BLOCK_DIM, BLOCK_DIM):
                r1 = r0 + BLOCK_DIM
                out[c1:acol,r0:r1] = a[r0:r1,c1:acol].T
            out[c1:acol,r1:arow] = a[r1:arow,c1:acol].T
            return out
        else:
            return a.transpose(axes)

    if a.ndim == 2:
        arow, acol = a.shape
        c_shape = (ctypes.c_int*3)(1, arow, acol)
        out = numpy.ndarray((acol, arow), a.dtype, buffer=out)
    elif a.ndim == 3 and axes == (0,2,1):
        d0, arow, acol = a.shape
        c_shape = (ctypes.c_int*3)(d0, arow, acol)
        out = numpy.ndarray((d0, acol, arow), a.dtype, buffer=out)
    else:
        raise NotImplementedError

    assert(a.flags.c_contiguous)
    if a.dtype == numpy.double:
        fn = _np_helper.NPdtranspose_021
    else:
        fn = _np_helper.NPztranspose_021
    fn.restype = ctypes.c_void_p
    fn(c_shape, a.ctypes.data_as(ctypes.c_void_p),
       out.ctypes.data_as(ctypes.c_void_p))
    return out

def transpose_sum(a, inplace=False, out=None):
    '''Computing a + a.T with better memory efficiency

    Examples:

    >>> transpose_sum(numpy.arange(4.).reshape(2,2))
    [[ 0.  3.]
     [ 3.  6.]]
    '''
    return hermi_sum(a, inplace=inplace, out=out)

def hermi_sum(a, axes=None, hermi=HERMITIAN, inplace=False, out=None):
    '''Computing a + a.T.conj() with better memory efficiency

    Examples:

    >>> transpose_sum(numpy.arange(4.).reshape(2,2))
    [[ 0.  3.]
     [ 3.  6.]]
    '''
    if inplace:
        out = a
    else:
        out = numpy.ndarray(a.shape, a.dtype, buffer=out)

    if (not a.flags.c_contiguous
        or (a.dtype != numpy.double and a.dtype != numpy.complex128)):
        if a.ndim == 2:
            na = a.shape[0]
            for c0, c1 in misc.prange(0, na, BLOCK_DIM):
                for r0, r1 in misc.prange(0, c0, BLOCK_DIM):
                    tmp = a[r0:r1,c0:c1] + a[c0:c1,r0:r1].conj().T
                    out[c0:c1,r0:r1] = tmp.T.conj()
                    out[r0:r1,c0:c1] = tmp
                # diagonal blocks
                tmp = a[c0:c1,c0:c1] + a[c0:c1,c0:c1].conj().T
                out[c0:c1,c0:c1] = tmp
            return out
        else:
            raise NotImplementedError('input array is not C-contiguous')

    if a.ndim == 2:
        assert(a.shape[0] == a.shape[1])
        c_shape = (ctypes.c_int*3)(1, a.shape[0], a.shape[1])
    elif a.ndim == 3 and axes == (0,2,1):
        assert(a.shape[1] == a.shape[2])
        c_shape = (ctypes.c_int*3)(*(a.shape))
    else:
        raise NotImplementedError

    assert(a.flags.c_contiguous)
    if a.dtype == numpy.double:
        fn = _np_helper.NPdsymm_021_sum
    else:
        fn = _np_helper.NPzhermi_021_sum
    fn(c_shape, a.ctypes.data_as(ctypes.c_void_p),
       out.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(hermi))
    return out

# NOTE: NOT assume array a, b to be C-contiguous, since a and b are two
# pointers we want to pass in.
# numpy.dot might not call optimized blas
def ddot(a, b, alpha=1, c=None, beta=0):
    '''Matrix-matrix multiplication for double precision arrays
    '''
    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[1]
    if a.flags.c_contiguous:
        trans_a = 'N'
    elif a.flags.f_contiguous:
        trans_a = 'T'
        a = a.T
    else:
        a = numpy.asarray(a, order='C')
        trans_a = 'N'
        #raise ValueError('a.flags: %s' % str(a.flags))

    assert(k == b.shape[0])
    if b.flags.c_contiguous:
        trans_b = 'N'
    elif b.flags.f_contiguous:
        trans_b = 'T'
        b = b.T
    else:
        b = numpy.asarray(b, order='C')
        trans_b = 'N'
        #raise ValueError('b.flags: %s' % str(b.flags))

    if c is None:
        c = numpy.empty((m,n))
        beta = 0
    else:
        assert(c.shape == (m,n))

    return _dgemm(trans_a, trans_b, m, n, k, a, b, c, alpha, beta)

def zdot(a, b, alpha=1, c=None, beta=0):
    '''Matrix-matrix multiplication for double complex arrays
    '''
    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[1]
    if a.flags.c_contiguous:
        trans_a = 'N'
    elif a.flags.f_contiguous:
        trans_a = 'T'
        a = a.T
    else:
        raise ValueError('a.flags: %s' % str(a.flags))

    assert(k == b.shape[0])
    if b.flags.c_contiguous:
        trans_b = 'N'
    elif b.flags.f_contiguous:
        trans_b = 'T'
        b = b.T
    else:
        raise ValueError('b.flags: %s' % str(b.flags))

    if c is None:
        beta = 0
        c = numpy.empty((m,n), dtype=numpy.complex128)
    else:
        assert(c.shape == (m,n))

    return _zgemm(trans_a, trans_b, m, n, k, a, b, c, alpha, beta)

def dot_cublas(a, b, alpha=1, c=None, beta=0):
    ab = cupy.dot(a, b)
    if alpha != 1:
        ab *= alpha
    if c is not None and beta != 0:
        assert isinstance(c, cupy.ndarray)
        c = c*beta + ab
    else:
        c = ab
    return c

def dot(a, b, alpha=1, c=None, beta=0):
    if CUBLAS and isinstance(a, cupy.ndarray) and isinstance(b, cupy.ndarray):
        return dot_cublas(a, b, alpha, c, beta)

    atype = a.dtype
    btype = b.dtype

    if atype == numpy.float64 and btype == numpy.float64:
        if c is None or c.dtype == numpy.float64:
            return ddot(a, b, alpha, c, beta)
        else:
            cr = numpy.asarray(c.real, order='C')
            c.real = ddot(a, b, alpha, cr, beta)
            return c

    elif atype == numpy.complex128 and btype == numpy.complex128:
        # Gauss's complex multiplication algorithm may affect numerical stability
        #k1 = ddot(a.real+a.imag, b.real.copy(), alpha)
        #k2 = ddot(a.real.copy(), b.imag-b.real, alpha)
        #k3 = ddot(a.imag.copy(), b.real+b.imag, alpha)
        #ab = k1-k3 + (k1+k2)*1j
        return zdot(a, b, alpha, c, beta)

    elif atype == numpy.float64 and btype == numpy.complex128:
        if b.flags.f_contiguous:
            order = 'F'
        else:
            order = 'C'
        cr = ddot(a, numpy.asarray(b.real, order=order), alpha)
        ci = ddot(a, numpy.asarray(b.imag, order=order), alpha)
        ab = numpy.ndarray(cr.shape, dtype=numpy.complex128, buffer=c)
        if c is None or beta == 0:
            ab.real = cr
            ab.imag = ci
        else:
            ab *= beta
            ab.real += cr
            ab.imag += ci
        return ab

    elif atype == numpy.complex128 and btype == numpy.float64:
        if a.flags.f_contiguous:
            order = 'F'
        else:
            order = 'C'
        cr = ddot(numpy.asarray(a.real, order=order), b, alpha)
        ci = ddot(numpy.asarray(a.imag, order=order), b, alpha)
        ab = numpy.ndarray(cr.shape, dtype=numpy.complex128, buffer=c)
        if c is None or beta == 0:
            ab.real = cr
            ab.imag = ci
        else:
            ab *= beta
            ab.real += cr
            ab.imag += ci
        return ab

    else:
        if c is None:
            c = numpy.dot(a, b) * alpha
        elif beta == 0:
            c[:] = numpy.dot(a, b) * alpha
        else:
            c *= beta
            c += numpy.dot(a, b) * alpha
        return c

# a, b, c in C-order
def _dgemm(trans_a, trans_b, m, n, k, a, b, c, alpha=1, beta=0,
           offseta=0, offsetb=0, offsetc=0):
    if a.size == 0 or b.size == 0:
        if beta == 0:
            c[:] = 0
        else:
            c[:] *= beta
        return c

    assert(a.flags.c_contiguous)
    assert(b.flags.c_contiguous)
    assert(c.flags.c_contiguous)

    _np_helper.NPdgemm(ctypes.c_char(trans_b.encode('ascii')),
                       ctypes.c_char(trans_a.encode('ascii')),
                       ctypes.c_int(n), ctypes.c_int(m), ctypes.c_int(k),
                       ctypes.c_int(b.shape[1]), ctypes.c_int(a.shape[1]),
                       ctypes.c_int(c.shape[1]),
                       ctypes.c_int(offsetb), ctypes.c_int(offseta),
                       ctypes.c_int(offsetc),
                       b.ctypes.data_as(ctypes.c_void_p),
                       a.ctypes.data_as(ctypes.c_void_p),
                       c.ctypes.data_as(ctypes.c_void_p),
                       ctypes.c_double(alpha), ctypes.c_double(beta))
    return c
def _zgemm(trans_a, trans_b, m, n, k, a, b, c, alpha=1, beta=0,
           offseta=0, offsetb=0, offsetc=0):
    if a.size == 0 or b.size == 0:
        if beta == 0:
            c[:] = 0
        else:
            c[:] *= beta
        return c

    assert(a.flags.c_contiguous)
    assert(b.flags.c_contiguous)
    assert(c.flags.c_contiguous)
    assert(a.dtype == numpy.complex128)
    assert(b.dtype == numpy.complex128)
    assert(c.dtype == numpy.complex128)

    _np_helper.NPzgemm(ctypes.c_char(trans_b.encode('ascii')),
                       ctypes.c_char(trans_a.encode('ascii')),
                       ctypes.c_int(n), ctypes.c_int(m), ctypes.c_int(k),
                       ctypes.c_int(b.shape[1]), ctypes.c_int(a.shape[1]),
                       ctypes.c_int(c.shape[1]),
                       ctypes.c_int(offsetb), ctypes.c_int(offseta),
                       ctypes.c_int(offsetc),
                       b.ctypes.data_as(ctypes.c_void_p),
                       a.ctypes.data_as(ctypes.c_void_p),
                       c.ctypes.data_as(ctypes.c_void_p),
                       (ctypes.c_double*2)(alpha.real, alpha.imag),
                       (ctypes.c_double*2)(beta.real, beta.imag))
    return c

def frompointer(pointer, count, dtype=float):
    '''Interpret a buffer that the pointer refers to as a 1-dimensional array.

    Args:
        pointer : int or ctypes pointer
            address of a buffer
        count : int
            Number of items to read.
        dtype : data-type, optional
            Data-type of the returned array; default: float.

    Examples:

    >>> s = numpy.ones(3, dtype=numpy.int32)
    >>> ptr = s.ctypes.data
    >>> frompointer(ptr, count=6, dtype=numpy.int16)
    [1, 0, 1, 0, 1, 0]
    '''
    dtype = numpy.dtype(dtype)
    count *= dtype.itemsize
    buf = (ctypes.c_char * count).from_address(pointer)
    a = numpy.ndarray(count, dtype=numpy.int8, buffer=buf)
    return a.view(dtype)

from distutils.version import LooseVersion
if LooseVersion(numpy.__version__) <= LooseVersion('1.6.0'):
    def norm(x, ord=None, axis=None):
        '''numpy.linalg.norm for numpy 1.6.*
        '''
        if axis is None or ord is not None:
            return numpy.linalg.norm(x, ord)
        else:
            x = numpy.asarray(x)
            axes = string.ascii_lowercase[:x.ndim]
            target = axes.replace(axes[axis], '')
            descr = '%s,%s->%s' % (axes, axes, target)
            xx = _numpy_einsum(descr, x.conj(), x)
            return numpy.sqrt(xx.real)
else:
    norm = numpy.linalg.norm
del(LooseVersion)

def cond(x, p=None):
    '''Compute the condition number'''
    if isinstance(x, numpy.ndarray) and x.ndim == 2 or p is not None:
        return numpy.linalg.cond(x, p)
    else:
        return numpy.asarray([numpy.linalg.cond(xi) for xi in x])

def cartesian_prod(arrays, out=None):
    '''
    Generate a cartesian product of input arrays.
    http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays

    Args:
        arrays : list of array-like
            1-D arrays to form the cartesian product of.
        out : ndarray
            Array to place the cartesian product in.

    Returns:
        out : ndarray
            2-D array of shape (M, len(arrays)) containing cartesian products
            formed of input arrays.

    Examples:

    >>> cartesian_prod(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    '''
    arrays = [numpy.asarray(x) for x in arrays]
    dtype = numpy.result_type(*arrays)
    nd = len(arrays)
    dims = [nd] + [len(x) for x in arrays]
    out = numpy.ndarray(dims, dtype, buffer=out)

    shape = [-1] + [1] * nd
    for i, arr in enumerate(arrays):
        out[i] = arr.reshape(shape[:nd-i])

    return out.reshape(nd,-1).T

def direct_sum(subscripts, *operands):
    '''Apply the summation over many operands with the einsum fashion.

    Examples:

    >>> a = numpy.random.random((6,5))
    >>> b = numpy.random.random((4,3,2))
    >>> direct_sum('ij,klm->ijklm', a, b).shape
    (6, 5, 4, 3, 2)
    >>> direct_sum('ij,klm', a, b).shape
    (6, 5, 4, 3, 2)
    >>> direct_sum('i,j,klm->mjlik', a[0], a[:,0], b).shape
    (2, 6, 3, 5, 4)
    >>> direct_sum('ij-klm->ijklm', a, b).shape
    (6, 5, 4, 3, 2)
    >>> direct_sum('ij+klm', a, b).shape
    (6, 5, 4, 3, 2)
    >>> direct_sum('-i-j+klm->mjlik', a[0], a[:,0], b).shape
    (2, 6, 3, 5, 4)
    >>> c = numpy.random((3,5))
    >>> z = direct_sum('ik+jk->kij', a, c).shape  # This is slow
    >>> abs(a.T.reshape(5,6,1) + c.reshape(5,1,3) - z).sum()
    0.0
    '''

    def sign_and_symbs(subscript):
        ''' sign list and notation list'''
        subscript = subscript.replace(' ', '').replace(',', '+')

        if subscript[0] not in '+-':
            subscript = '+' + subscript
        sign = [x for x in subscript if x in '+-']

        symbs = subscript[1:].replace('-', '+').split('+')
        #s = ''.join(symbs)
        #assert(len(set(s)) == len(s))  # make sure no duplicated symbols
        return sign, symbs

    if '->' in subscripts:
        src, dest = subscripts.split('->')
        sign, src = sign_and_symbs(src)
        dest = dest.replace(' ', '')
    else:
        sign, src = sign_and_symbs(subscripts)
        dest = ''.join(src)
    assert(len(src) == len(operands))

    for i, symb in enumerate(src):
        op = numpy.asarray(operands[i])
        assert(len(symb) == op.ndim)
        unisymb = set(symb)
        if len(unisymb) != len(symb):
            unisymb = ''.join(unisymb)
            op = _numpy_einsum('->'.join((symb, unisymb)), op)
            src[i] = unisymb
        if i == 0:
            if sign[i] == '+':
                out = op
            else:
                out = -op
        elif sign[i] == '+':
            out = out.reshape(out.shape+(1,)*op.ndim) + op
        else:
            out = out.reshape(out.shape+(1,)*op.ndim) - op

    out = _numpy_einsum('->'.join((''.join(src), dest)), out)
    out.flags.writeable = True  # old numpy has this issue
    return out

def condense(opname, a, loc_x, loc_y=None):
    '''
    .. code-block:: python

        for i,i0 in enumerate(loc_x):
            i1 = loc_x[i+1]
            for j,j0 in enumerate(loc_y):
                j1 = loc_y[j+1]
                out[i,j] = op(a[i0:i1,j0:j1])
    '''
    assert(a.dtype == numpy.double)
    if not opname.startswith('NP_'):
        opname = 'NP_' + opname
    op = getattr(_np_helper, opname)
    if loc_y is None:
        loc_y = loc_x
    loc_x = numpy.asarray(loc_x, numpy.int32)
    loc_y = numpy.asarray(loc_y, numpy.int32)
    nloc_x = loc_x.size - 1
    nloc_y = loc_y.size - 1
    if a.flags.f_contiguous:
        out = numpy.zeros((nloc_x, nloc_y), order='F')
    else:
        a = numpy.asarray(a, order='C')
        out = numpy.zeros((nloc_x, nloc_y))
    _np_helper.NPcondense(op, out.ctypes.data_as(ctypes.c_void_p),
                          a.ctypes.data_as(ctypes.c_void_p),
                          loc_x.ctypes.data_as(ctypes.c_void_p),
                          loc_y.ctypes.data_as(ctypes.c_void_p),
                          ctypes.c_int(nloc_x), ctypes.c_int(nloc_y))
    return out

def expm(a):
    '''Equivalent to scipy.linalg.expm'''
    bs = [a.copy()]
    n = 0
    for n in range(1, 14):
        bs.append(ddot(bs[-1], a))
        radius = (2**(n*(n+2))*math.factorial(n+2)*1e-16) **((n+1.)/(n+2))
        #print(n, radius, bs[-1].max(), -bs[-1].min())
        if bs[-1].max() < radius and -bs[-1].min() < radius:
            break

    y = numpy.eye(a.shape[0])
    fac = 1
    for i, b in enumerate(bs):
        fac *= i + 1
        b *= (.5**(n*(i+1)) / fac)
        y += b
    buf, bs = bs[0], None
    for i in range(n):
        ddot(y, y, 1, buf, 0)
        y, buf = buf, y
    return y

def exp(a, out=None, **kwargs):
    '''
    Multi-threaded numpy.exp
    '''
    if not (a.flags.c_contiguous or a.flags.f_contiguous):
        return numpy.exp(a, out=out, **kwargs)

    dtype = a.dtype
    if dtype == numpy.double:
        fn = getattr(_np_helper, "NPdexp", None)
    elif dtype == numpy.complex128:
        fn = getattr(_np_helper, "NPzexp", None)
    else:
        return numpy.exp(a, out=out, **kwargs)

    if out is None:
        out = numpy.empty_like(a)
    else:
        if (out.dtype != a.dtype or
            not (out.flags.c_contiguous or out.flags.f_contiguous)):
            return numpy.exp(a, out=out, **kwargs)

    if a.flags.c_contiguous:
        a = numpy.asarray(a, order='C')
        out = numpy.asarray(out, order='C')
    elif a.flags.f_contiguous:
        a = numpy.asarray(a, order='F')
        out = numpy.asarray(out, order='F')

    n = a.size
    fn(out.ctypes.data_as(ctypes.c_void_p),
       a.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(n))
    return out

def cos(a, out=None, **kwargs):
    '''
    Multi-threaded numpy.cos
    '''
    if not (a.flags.c_contiguous or a.flags.f_contiguous):
        return numpy.cos(a, out=out, **kwargs)
    fn = getattr(_np_helper, "NPcos", None)
    if out is None:
        out = numpy.empty_like(a)
    else:
        if (out.dtype != a.dtype or
            not (out.flags.c_contiguous or out.flags.f_contiguous)):
            return numpy.exp(a, out=out, **kwargs)
    if a.flags.c_contiguous:
        a = numpy.asarray(a, order='C')
        out = numpy.asarray(out, order='C')
    elif a.flags.f_contiguous:
        a = numpy.asarray(a, order='F')
        out = numpy.asarray(out, order='F')
    n = a.size
    fn(out.ctypes.data_as(ctypes.c_void_p),
       a.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(n))
    return out

def sin(a, out=None, **kwargs):
    '''
    Multi-threaded numpy.sin
    '''
    if not (a.flags.c_contiguous or a.flags.f_contiguous):
        return numpy.cos(a, out=out, **kwargs)
    fn = getattr(_np_helper, "NPsin", None)
    if out is None:
        out = numpy.empty_like(a)
    else:
        if (out.dtype != a.dtype or
            not (out.flags.c_contiguous or out.flags.f_contiguous)):
            return numpy.exp(a, out=out, **kwargs)
    if a.flags.c_contiguous:
        a = numpy.asarray(a, order='C')
        out = numpy.asarray(out, order='C')
    elif a.flags.f_contiguous:
        a = numpy.asarray(a, order='F')
        out = numpy.asarray(out, order='F')
    n = a.size
    fn(out.ctypes.data_as(ctypes.c_void_p),
       a.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(n))
    return out

def sum(a, axis=-1, dtype=None, out=None, **kwargs):
    '''
    Multi-threaded numpy.sum
    '''
    if not (a.flags.c_contiguous or a.flags.f_contiguous):
        return numpy.sum(a, axis, dtype, out, **kwargs)
    if dtype is None:
        dtype = a.dtype
    if dtype != a.dtype:
        return numpy.sum(a, axis, dtype, out, **kwargs)

    if dtype == numpy.double:
        fn = getattr(_np_helper, "NPdsum", None)
    elif dtype == numpy.complex128:
        fn = getattr(_np_helper, "NPzsum", None)
    else:
        return numpy.sum(a, axis, dtype, out, **kwargs)

    if axis == len(a.shape)-1:
        axis = -1
    elif axis + len(a.shape) == 0:
        axis = 0

    if axis == -1 and a.flags.c_contiguous:
        lda = a.shape[-1]
        a = numpy.asarray(a, order='C')
        out_shape = tuple(a.shape[:-1])
        if out is None:
            out = numpy.empty(out_shape, order='C', dtype=dtype)
        else:
            if ((numpy.asarray(out.shape) - numpy.asarray(out_shape)).sum() != 0 or
                (not out.flags.c_contiguous) or out.dtype != dtype):
                return numpy.sum(a, axis, dtype, out, **kwargs)
    elif axis == 0 and a.flags.f_contiguous:
        lda = a.shape[0]
        a = numpy.asarray(a, order='F')
        out_shape = tuple(a.shape[1:])
        if out is None:
            out = numpy.empty(out_shape, order='F', dtype=dtype)
        else:
            if ((numpy.asarray(out.shape) - numpy.asarray(out_shape)).sum() != 0 or
                (not out.flags.f_contiguous) or out.dtype != dtype):
                return numpy.sum(a, axis, dtype, out, **kwargs)
    else:
        return numpy.sum(a, axis, dtype, out, **kwargs)

    fn(out.ctypes.data_as(ctypes.c_void_p),
       a.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(lda), ctypes.c_int(out.size))
    return out

def add(a, b, out=None, **kwargs):
    if (a.shape != b.shape or a.dtype != b.dtype or
        not(a.flags.c_contiguous or a.flags.f_contiguous) or
        not(b.flags.c_contiguous or b.flags.f_contiguous)):
        return numpy.add(a, b, out, **kwargs)

    if a.dtype == numpy.double:
        fn = getattr(_np_helper, "NPdadd", None)
    elif a.dtype == numpy.complex128:
        fn = getattr(_np_helper, "NPzadd", None)
    else:
        return numpy.add(a, b, out, **kwargs)

    if out is None:
        if a.flags.c_contiguous:
            out = numpy.empty_like(a, order='C')
        else:
            out = numpy.empty_like(a, order='F')
    else:
        if (a.shape != out.shape or
            not (out.flags.c_contiguous or out.flags.f_contiguous)):
            return numpy.add(a, b, out, **kwargs)

    if a.flags.c_contiguous:
        if not out.flags.c_contiguous:
            return numpy.add(a, b, out, **kwargs)
        b = numpy.asarray(b, order='C')
    else:
        if not out.flags.f_contiguous:
            return numpy.add(a, b, out, **kwargs)
        b = numpy.asarray(b, order='F')

    fn(out.ctypes.data_as(ctypes.c_void_p),
       a.ctypes.data_as(ctypes.c_void_p),
       b.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(out.size))
    return out

def subtract(a, b, out=None, **kwargs):
    if (a.shape != b.shape or a.dtype != b.dtype or
        not(a.flags.c_contiguous or a.flags.f_contiguous) or
        not(b.flags.c_contiguous or b.flags.f_contiguous)):
        return numpy.subtract(a, b, out, **kwargs)

    if a.dtype == numpy.double:
        fn = getattr(_np_helper, "NPdsubtract", None)
    elif a.dtype == numpy.complex128:
        fn = getattr(_np_helper, "NPzsubtract", None)
    else:
        return numpy.subtract(a, b, out, **kwargs)

    if out is None:
        if a.flags.c_contiguous:
            out = numpy.empty_like(a, order='C')
        else:
            out = numpy.empty_like(a, order='F')
    else:
        if (a.shape != out.shape or
            not (out.flags.c_contiguous or out.flags.f_contiguous)):
            return numpy.subtract(a, b, out, **kwargs)

    if a.flags.c_contiguous:
        if not out.flags.c_contiguous:
            return numpy.subtract(a, b, out, **kwargs)
        b = numpy.asarray(b, order='C')
    else:
        if not out.flags.f_contiguous:
            return numpy.subtract(a, b, out, **kwargs)
        b = numpy.asarray(b, order='F')

    fn(out.ctypes.data_as(ctypes.c_void_p),
       a.ctypes.data_as(ctypes.c_void_p),
       b.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(out.size))
    return out

def multiply(a, b, out=None, **kwargs):
    if numpy.isscalar(a) or numpy.isscalar(b):
        dtype = numpy.result_type(a,b)

        number = a if numpy.isscalar(a) else b
        matrix = b if numpy.isscalar(a) else a
        if not (matrix.flags.c_contiguous or matrix.flags.f_contiguous):
            return numpy.multiply(a, b, out=out, **kwargs)

        number = numpy.asarray([number,], dtype=dtype)
        matrix = numpy.asarray(matrix, dtype=dtype)
        if dtype == numpy.double:
            fn = getattr(_np_helper, "NPdmultiply_scalar", None)
        elif dtype == numpy.complex128:
            fn = getattr(_np_helper, "NPzmultiply_scalar", None)
        else:
            return numpy.multiply(a, b, out=out, **kwargs)

        if out is None:
            out = numpy.empty_like(matrix)
        else:
            assert out.dtype == dtype
            assert out.size == matrix.size

        fn(out.ctypes.data_as(ctypes.c_void_p),
           matrix.ctypes.data_as(ctypes.c_void_p),
           number.ctypes.data_as(ctypes.c_void_p),
           ctypes.c_size_t(out.size))
        return out

    if (a.shape != b.shape or
        not(a.flags.c_contiguous or a.flags.f_contiguous) or
        not(b.flags.c_contiguous or b.flags.f_contiguous)):
        return numpy.multiply(a, b, out=out, **kwargs)

    if a.dtype == b.dtype:
        if a.dtype == numpy.double:
            fn = getattr(_np_helper, "NPdmultiply", None)
        elif a.dtype == numpy.complex128:
            fn = getattr(_np_helper, "NPzmultiply", None)
        else:
            return numpy.multiply(a, b, out=out, **kwargs)
    else:
        if (a.dtype == numpy.double and b.dtype == numpy.complex128):
            fn = getattr(_np_helper, "NPmultiply_dz", None)
        elif (b.dtype == numpy.double and a.dtype == numpy.complex128):
            fn = getattr(_np_helper, "NPmultiply_zd", None)
        else:
            return numpy.multiply(a, b, out=out, **kwargs)

    if out is None:
        if a.flags.c_contiguous:
            out = numpy.empty_like(a, order='C')
        else:
            out = numpy.empty_like(a, order='F')
    else:
        if (a.shape != out.shape or
            not (out.flags.c_contiguous or out.flags.f_contiguous)):
            return numpy.multiply(a, b, out=out, **kwargs)

    if a.flags.c_contiguous:
        if not out.flags.c_contiguous or not b.flags.c_contiguous:
            return numpy.multiply(a, b, out=out, **kwargs)
    else:
        if not out.flags.f_contiguous or not b.flags.f_contiguous:
            return numpy.multiply(a, b, out=out, **kwargs)

    fn(out.ctypes.data_as(ctypes.c_void_p),
       a.ctypes.data_as(ctypes.c_void_p),
       b.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(out.size))
    return out

def multiply_sum(a, b, axis=0):
    # out = einsum("ij,ij->j", a, b) if axis == 0
    # out = einsum("ij,ij->i", a, b) if axis == 1
    dtype = numpy.result_type(a,b)
    a = numpy.asarray(a, order='C', dtype=dtype)
    b = numpy.asarray(b, order='C', dtype=dtype)
    if axis == -1:
        ncol = a.shape[-1]
        out_shape = a.shape[:(a.ndim-1)]
        out = numpy.zeros(tuple(out_shape), order='C', dtype=dtype)
        nrow = out.size
    elif a.ndim == 2:
        nrow, ncol = a.shape
        if axis == 0:
            out = numpy.zeros((ncol,), order='C', dtype=dtype)
        else:
            out = numpy.zeros((nrow,), order='C', dtype=dtype)
    else:
        raise NotImplementedError
    assert(a.flags.c_contiguous)
    assert(b.flags.c_contiguous)
    assert(out.flags.c_contiguous)

    if dtype == numpy.double:
        fn = getattr(_np_helper, "NPdmultiplysum", None)
    elif dtype == numpy.complex128:
        fn = getattr(_np_helper, "NPzmultiplysum", None)
    else:
        raise TypeError
    fn(out.ctypes.data_as(ctypes.c_void_p),
       a.ctypes.data_as(ctypes.c_void_p),
       b.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(nrow), ctypes.c_int(ncol), ctypes.c_int(axis))
    return out

def reciprocal(x, out=None, order='K', dtype=None, **kwargs):
    if (x.dtype not in [float, numpy.double, numpy.complex128]
        or order != 'K' or not (x.flags.c_contiguous or x.flags.f_contiguous)):
        return numpy.reciprocal(x, out=out, order=order, dtype=dtype, **kwargs)

    dtype = numpy.result_type(x.dtype, dtype)
    if out is None:
        out = numpy.empty_like(x, dtype=dtype)
    else:
        assert out.dtype == dtype
        assert out.size == x.size

    if x.dtype == numpy.complex128:
        fn = getattr(_np_helper, "NPzreciprocal", None)
    elif x.dtype == numpy.double:
        if dtype == numpy.complex128:
            fn = getattr(_np_helper, "NPreciprocal_d2z", None)
        else:
            fn = getattr(_np_helper, "NPdreciprocal", None)
    else:
        raise NotImplementedError

    fn(out.ctypes.data_as(ctypes.c_void_p),
       x.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_size_t(x.size))
    return out

def concatenate(arrays, axis=0, out=None, dtype=None, casting="same_kind"):
    use_numpy = False

    if dtype is None:
        dtype = numpy.result_type(*arrays)

    if dtype not in [float, numpy.double, numpy.complex128]:
        use_numpy = True

    c_contiguous = True
    for array in arrays:
        if array.dtype != dtype:
            use_numpy = True
        c_contiguous = c_contiguous and array.flags.c_contiguous

    if not c_contiguous:
        use_numpy = True

    ndim = arrays[0].ndim
    if axis == -1:
        axis = ndim - 1
    if axis == ndim - 1:
        use_numpy = True

    if use_numpy:
        return numpy.concatenate(arrays, axis=axis, out=out, dtype=dtype, casting=casting)

    shape = []
    pointer = []
    for array in arrays:
        shape.append(array.shape)
        pointer.append(array.ctypes.data)
    shape = numpy.asarray(shape, dtype=numpy.int32)
    pointer = numpy.asarray(pointer, dtype=numpy.uintp).ctypes.data_as(ctypes.c_void_p)
    out_shape = shape[0].copy()
    out_shape[axis] = shape[:,axis].sum()

    if out is None:
        out = numpy.empty(out_shape, dtype=dtype)
    else:
        assert (numpy.asarray(out.shape) - out_shape).sum() == 0
        assert out.dtype == dtype

    if dtype == numpy.double:
        fn = getattr(_np_helper, "NPdconcatenate", None)
    elif dtype == numpy.complex128:
        fn = getattr(_np_helper, "NPzconcatenate", None)
    else:
        raise NotImplementedError

    fn(out.ctypes.data_as(ctypes.c_void_p),
       pointer,
       ctypes.c_int(len(arrays)),
       out_shape.ctypes.data_as(ctypes.c_void_p),
       shape.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(ndim), ctypes.c_int(axis))

    return out

def copy(a, order='K', subok=False, dtype=None, out=None):
    if (subok or a.dtype not in [float, numpy.double, numpy.complex128]
        or order != 'K' or not (a.flags.c_contiguous or a.flags.f_contiguous)):
        return numpy.copy(numpy.asarray(a, dtype=dtype), order=order, subok=subok)

    #dtype = numpy.result_type(a.dtype, dtype)

    if out is not None:
        assert out.dtype == numpy.result_type(dtype, out.dtype)
        assert out.size == a.size
    else:
        out = numpy.empty_like(a, dtype=dtype)

    if a.dtype == numpy.complex128:
        if dtype == numpy.complex128:
            fn = getattr(_np_helper, "NPzcopy_omp")
        else:
            fn = getattr(_np_helper, "NPcopy_z2d")
    elif a.dtype == numpy.double:
        if dtype == numpy.complex128:
            fn = getattr(_np_helper, "NPcopy_d2z")
        else:
            fn = getattr(_np_helper, "NPdcopy_omp")
    else:
        raise NotImplementedError

    fn(out.ctypes.data_as(ctypes.c_void_p),
       a.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_size_t(a.size))
    return out

def vdot(a, b):
    if a.ndim > 1 or b.ndim > 1:
        return numpy.vdot(a,b)

    if (not (a.flags.c_contiguous or a.flags.f_contiguous) or
        not (b.flags.c_contiguous or b.flags.f_contiguous)):
        return numpy.vdot(a,b)

    if a.dtype == b.dtype and a.size == b.size:
        if a.dtype == numpy.complex128:
            fn = getattr(_np_helper, "NPzvdot", None)
        elif a.dtype == numpy.double:
            fn = getattr(_np_helper, "NPdvdot", None)
        else:
            return numpy.vdot(a,b)
    else:
        return numpy.vdot(a,b)

    out = numpy.zeros((1,), dtype=a.dtype)
    fn(out.ctypes.data_as(ctypes.c_void_p),
       a.ctypes.data_as(ctypes.c_void_p),
       b.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_size_t(a.size))
    return out[0]

class NPArrayWithTag(numpy.ndarray):
    # Initialize kwargs in function tag_array
    #def __new__(cls, a, **kwargs):
    #    obj = numpy.asarray(a).view(cls)
    #    obj.__dict__.update(kwargs)
    #    return obj

    # Customize __reduce__ and __setstate__ to keep tags after serialization
    # pickle.loads(pickle.dumps(tagarray)).  This is needed by mpi communication
    def __reduce__(self):
        pickled = numpy.ndarray.__reduce__(self)
        state = pickled[2] + (self.__dict__,)
        return (pickled[0], pickled[1], state)

    def __setstate__(self, state):
        numpy.ndarray.__setstate__(self, state[0:-1])
        self.__dict__.update(state[-1])

    # Whenever the contents of the array was modified (through ufunc), the tag
    # should be expired. Overwrite the output of ufunc to restore ndarray type.
    def __array_wrap__(self, out, context=None):
        return numpy.ndarray.__array_wrap__(self, out, context).view(numpy.ndarray)


def tag_array(a, **kwargs):
    '''Attach attributes to numpy ndarray. The attribute name and value are
    obtained from the keyword arguments.
    '''
    # Make a shadow copy in any circumstance by converting it to an nparray.
    # If a is an object of NPArrayWithTag, all attributes will be lost in this
    # conversion. They need to be restored.
    t = numpy.asarray(a).view(NPArrayWithTag)

    if isinstance(a, NPArrayWithTag):
        t.__dict__.update(a.__dict__)
    t.__dict__.update(kwargs)
    return t

#TODO: merge with function pbc.cc.kccsd_rhf.vector_to_nested
def split_reshape(a, shapes):
    '''
    Split a vector into multiple tensors. shapes is a list of tuples.
    The entries of shapes indicate the shape of each tensor.

    Returns:
        tensors : a list of tensors

    Examples:

    >>> a = numpy.arange(12)
    >>> split_reshape(a, ((2,3), (1,), ((2,2), (1,1))))
    [array([[0, 1, 2],
            [3, 4, 5]]),
     array([6]),
     [array([[ 7,  8],
             [ 9, 10]]),
      array([[11]])]]
    '''
    if isinstance(shapes[0], (int, numpy.integer)):
        return a.reshape(shapes)

    def sub_split(a, shapes):
        tensors = []
        p1 = 0
        for shape in shapes:
            if isinstance(shape[0], (int, numpy.integer)):
                p0, p1 = p1, p1 + numpy.prod(shape)
                tensors.append(a[p0:p1].reshape(shape))
            else:
                subtensors, size = sub_split(a[p1:], shape)
                p1 += size
                tensors.append(subtensors)
        size = p1
        return tensors, size
    return sub_split(a, shapes)[0]

if __name__ == '__main__':
    a = numpy.random.random((30,40,5,10))
    b = numpy.random.random((10,30,5,20))
    c = numpy.random.random((10,20,20))
    d = numpy.random.random((20,10))
    f = einsum('ijkl,xiky,ayp,px->ajl', a,b,c,d, optimize=True)
    ref = einsum('ijkl,xiky->jlxy', a, b)
    ref = einsum('jlxy,ayp->jlxap', ref, c)
    ref = einsum('jlxap,px->ajl', ref, d)
    print(abs(ref-f).max())
