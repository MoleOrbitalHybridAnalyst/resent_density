#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

import unittest
import numpy as np
from pyscf.lib.misc import finger
from pyscf.pbc import scf, gto, grad

cell = gto.Cell()
a = 3.370137329
cell.atom= [['C', [0.0, 0.0, 0.0]], ['C', [.5*a, .5*a, .5*a + 0.01]]]
cell.a = np.asarray([[0, a, a], [a, 0, a], [a, a, 0]])
cell.basis = 'gth-szv'
cell.verbose= 4
cell.pseudo = 'gth-pade'
cell.unit = 'bohr'
cell.build()

kpts = cell.make_kpts([1,1,2])
disp = 1e-5

def tearDownModule():
    global cell, a, kpts, disp
    cell.stdout.close()
    del cell, a, kpts, disp


class KnownValues(unittest.TestCase):
    def test_krhf_grad(self):
        g_scan = scf.KRHF(cell, kpts, exxdiv=None).nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        self.assertAlmostEqual(finger(g), -0.07021772172215038, 7)

        mfs = g_scan.base.as_scanner()
        e1 = mfs([['C', [0.0, 0.0, 0.0]], ['C', [.5*a, .5*a, .5*a + 0.01 + disp/2.0]]])
        e2 = mfs([['C', [0.0, 0.0, 0.0]], ['C', [.5*a, .5*a, .5*a + 0.01 - disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, 7)

    def test_grad_nuc(self):
        gnuc = grad.krhf.grad_nuc(cell)
        gref = np.asarray([[0, 0, -8.75413236e-03],
                           [0, 0, 8.75413236e-03]])
        self.assertAlmostEqual(abs(gnuc-gref).max(), 0, 9)

if __name__ == "__main__":
    print("Full Tests for KRHF Gradients")
    unittest.main()
