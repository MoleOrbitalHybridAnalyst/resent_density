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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
C extensions and helper functions
'''

from pyscf.lib import parameters
param = parameters
from pyscf.lib import numpy_helper
from pyscf.lib import linalg_helper
linalg = linalg_helper
from pyscf.lib import scipy_helper
from pyscf.lib import logger
from pyscf.lib import misc
from pyscf.lib.misc import *
from pyscf.lib.numpy_helper import *
from pyscf.lib.linalg_helper import *
from pyscf.lib.scipy_helper import *
from pyscf.lib import chkfile
from pyscf.lib import diis
from pyscf.lib.misc import StreamObject

# TODO following is temporary
try:
    from threadpoolctl import ThreadpoolController
except ImportError:
    class _ThreadpoolLimiter():
        def __init__(self, controller, *, limits=None, user_api=None):
            pass
        def __enter__(self):
            return self
        def __exit__(self, type, value, traceback):
            pass

    class ThreadpoolController():
        def __init__(self):
            pass
        def limit(self, *, limits=None, user_api=None):
            return _ThreadpoolLimiter(self, limits=limits, user_api=user_api)

threadpool_controller = ThreadpoolController()
