#
# Adapted from ASE socketio.py
# Author: Chenghan Li 
#

import socket
import numpy as np


def actualunixsocketname(name):
    return '/tmp/ipi_{}'.format(name)


class SocketClosed(OSError):
    pass


class IPIProtocol:
    """Communication using IPI protocol."""

    def __init__(self, socket, txt=None):
        self.socket = socket

        if txt is None:
            def log(*args):
                pass
        else:
            def log(*args):
                print('Driver:', *args, file=txt)
                txt.flush()
        self.log = log

    def sendmsg(self, msg):
        self.log('  sendmsg', repr(msg))
        # assert msg in self.statements, msg
        msg = msg.encode('ascii').ljust(12)
        self.socket.sendall(msg)

    def _recvall(self, nbytes):
        """Repeatedly read chunks until we have nbytes.

        Normally we get all bytes in one read, but that is not guaranteed."""
        remaining = nbytes
        chunks = []
        while remaining > 0:
            chunk = self.socket.recv(remaining)
            if len(chunk) == 0:
                # (If socket is still open, recv returns at least one byte)
                raise SocketClosed()
            chunks.append(chunk)
            remaining -= len(chunk)
        msg = b''.join(chunks)
        assert len(msg) == nbytes and remaining == 0
        return msg

    def recvmsg(self):
        msg = self._recvall(12)
        if not msg:
            raise SocketClosed()

        assert len(msg) == 12, msg
        msg = msg.rstrip().decode('ascii')
        # assert msg in self.responses, msg
        self.log('  recvmsg', repr(msg))
        return msg

    def send(self, a, dtype):
        buf = np.asarray(a, dtype).tobytes()
        # self.log('  send {}'.format(np.array(a).ravel().tolist()))
        self.log('  send {} bytes of {}'.format(len(buf), dtype))
        self.socket.sendall(buf)

    def recv(self, shape, dtype):
        a = np.empty(shape, dtype)
        nbytes = np.dtype(dtype).itemsize * np.prod(shape)
        buf = self._recvall(nbytes)
        assert len(buf) == nbytes, (len(buf), nbytes)
        self.log('  recv {} bytes of {}'.format(len(buf), dtype))
        # print(np.frombuffer(buf, dtype=dtype))
        a.flat[:] = np.frombuffer(buf, dtype=dtype)
        # self.log('  recv {}'.format(a.ravel().tolist()))
        assert np.isfinite(a).all()
        return a

    def recvposdata(self):
        cell = self.recv((3, 3), np.float64).T.copy()
        icell = self.recv((3, 3), np.float64).T.copy()
        natoms = self.recv(1, np.int32)
        natoms = int(natoms)
        positions = self.recv((natoms, 3), np.float64)
        return cell, icell, positions

    def sendforce(self, energy, forces, virial,
                  morebytes=np.zeros(1, dtype=np.byte)):
        assert np.array([energy]).size == 1
        assert forces.shape[1] == 3
        assert virial.shape == (3, 3)

        self.log(' sendforce')
        self.sendmsg('FORCEREADY')  # mind the units
        self.send(np.array([energy]), np.float64)
        natoms = len(forces)
        self.send(np.array([natoms]), np.int32)
        self.send(forces, np.float64)
        self.send(virial.T, np.float64)
        # We prefer to always send at least one byte due to trouble with
        # empty messages.  Reading a closed socket yields 0 bytes
        # and thus can be confused with a 0-length bytestring.
        self.send(np.array([len(morebytes)]), np.int32)
        self.send(morebytes, np.byte)

    def status(self):
        self.log(' status')
        self.sendmsg('STATUS')
        msg = self.recvmsg()
        return msg

    def end(self):
        self.log(' end')
        self.sendmsg('EXIT')

#    def recvinit(self):
#        self.log(' recvinit')
#        bead_index = self.recv(1, np.int32)
#        nbytes = self.recv(1, np.int32)
#        initbytes = self.recv(nbytes, np.byte)
#        return bead_index, initbytes
#
#    def sendinit(self):
#        # XXX Not sure what this function is supposed to send.
#        # It 'works' with QE, but for now we try not to call it.
#        self.log(' sendinit')
#        self.sendmsg('INIT')
#        self.send(0, np.int32)  # 'bead index' always zero for now
#        # We send one byte, which is zero, since things may not work
#        # with 0 bytes.  Apparently implementations ignore the
#        # initialization string anyway.
#        self.send(1, np.int32)
#        self.send(np.zeros(1), np.byte)  # initialization string


# @contextmanager
# def bind_unixsocket(socketfile):
#    assert socketfile.startswith('/tmp/ipi_'), socketfile
#    serversocket = socket.socket(socket.AF_UNIX)
#    try:
#        serversocket.bind(socketfile)
#    except OSError as err:
#        raise OSError('{}: {}'.format(err, repr(socketfile)))
#
#    try:
#        with serversocket:
#            yield serversocket
#    finally:
#        os.unlink(socketfile)
#
#
# @contextmanager
# def bind_inetsocket(port):
#    serversocket = socket.socket(socket.AF_INET)
#    serversocket.setsockopt(socket.SOL_SOCKET,
#                            socket.SO_REUSEADDR, 1)
#    serversocket.bind(('', port))
#    with serversocket:
#        yield serversocket

class SocketClient:
    default_port = 31415

    def __init__(self, host='localhost', port=None,
                 unixsocket=None, timeout=None, log=None):
        """Create client and connect to server.

        Parameters:

        host: string
            Hostname of server.  Defaults to localhost
        port: integer or None
            Port to which to connect.  By default 31415.
        unixsocket: string or None
            If specified, use corresponding UNIX socket.
            See documentation of unixsocket for SocketIOCalculator.
        timeout: float or None
            See documentation of timeout for SocketIOCalculator.
        log: file object or None
            Log events to this file """

        if unixsocket is not None:
            sock = socket.socket(socket.AF_UNIX)
            actualsocket = actualunixsocketname(unixsocket)
            sock.connect(actualsocket)
        else:
            if port is None:
                port = default_port
            sock = socket.socket(socket.AF_INET)
            sock.connect((host, port))
        sock.settimeout(timeout)
        self.host = host
        self.port = port
        self.unixsocket = unixsocket

        self.protocol = IPIProtocol(sock, txt=log)
        self.log = self.protocol.log
        self.closed = False

        self.bead_index = 0
        self.bead_initbytes = b''
        self.state = 'READY'


    def close(self):
        if not self.closed:
            self.log('Close SocketClient')
            self.closed = True
            self.protocol.socket.close()

    def calculate(self, atoms, use_stress):

        energy, forces, virial = atoms.get_efv()

        if use_stress:
            assert virial is not None
        else:
            virial = np.zeros((3,3))

        return energy, forces, virial

    def irun(self, atoms, use_stress=None):
        if use_stress is None:
            use_stress = False

        # For every step we either calculate or quit.  We need to
        # tell other MPI processes (if this is MPI-parallel) whether they
        # should calculate or quit.
        try:
            while True:
                try:
                    msg = self.protocol.recvmsg()
                except SocketClosed:
                    # Server closed the connection, but we want to
                    # exit gracefully anyway
                    msg = 'EXIT'

                if msg == 'EXIT':
                    return
                elif msg == 'STATUS':
                    self.protocol.sendmsg(self.state)
                elif msg == 'POSDATA':
                    assert self.state == 'READY'
                    cell, icell, positions = self.protocol.recvposdata()
                    atoms.set_positions_box(positions, cell)

                    energy, forces, virial = self.calculate(atoms, use_stress)

                    self.state = 'HAVEDATA'
                    # @@@@@@@@@@@@@@@
                    #self.log("positions")
                    #for pos in positions*0.52917721092:
                    #    self.log(*pos)
                    # @@@@@@@@@@@@@@@
                    yield
                elif msg == 'GETFORCE':
                    assert self.state == 'HAVEDATA', self.state
                    self.protocol.sendforce(energy, forces, virial)
                    self.state = 'READY'
                    # @@@@@@@@@@@@@@@
                    #self.log("energy", energy)
                    #self.log("forces")
                    #for frc in -forces:
                    #    self.log(*frc)
                    # @@@@@@@@@@@@@@@
#                elif msg == 'INIT':
#                    assert self.state == 'NEEDINIT'
#                    bead_index, initbytes = self.protocol.recvinit()
#                    self.bead_index = bead_index
#                    self.bead_initbytes = initbytes
#                    self.state = 'READY'
                else:
                    raise KeyError('Bad message', msg)
        finally:
            self.close()

    def run(self, atoms, use_stress=False):
        for _ in self.irun(atoms, use_stress=use_stress):
            pass


class SocketClients:

    def __init__(self, hosts=None, ports=None,
                 unixsockets=None, timeouts=None, paces=None,
                 logs=None):

        socks = list()

        if hosts is None:
            assert unixsockets is not None
            hosts = ['localhost'] * len(unixsockets)

        if unixsockets is not None:
            assert len(hosts) == len(unixsockets)
            for unixsocket in unixsockets:
                sock = socket.socket(socket.AF_UNIX)
                actualsocket = actualunixsocketname(unixsocket)
                sock.connect(actualsocket)
                socks.append(sock)
        else:
            assert ports is not None
            assert len(ports) == len(hosts)
            for host, port in zip(hosts, ports):
                sock = socket.socket(socket.AF_INET)
                sock.connect((host, port))
                socks.append(sock)
        if timeouts is None:
            timeouts = [None] * len(hosts)
        for sock, timeout in zip(socks, timeouts):
            sock.settimeout(timeout)

        if paces is None:
            paces = [1] * len(hosts)
        if logs is None:
            paces = [None] * len(hosts)

        self.hosts = hosts
        self.ports = ports
        self.unixsockets = unixsockets
        self.paces = paces

        self.protocols = [IPIProtocol(sock, txt=log) \
                for sock,log in zip(socks,logs)]
        self.logs = [p.log for p in self.protocols]
        self.closed = [False] * len(hosts)

        self.states = ['READY'] * len(hosts)

        self.istep = 0

    def close(self):
        for i in len(self.hosts):
            if not self.closed[i]:
                self.logs[i](f'Close SocketClient {i}')
                self.closed[i] = True
                self.protocols[i].socket.close()

    def calculate(self, atoms, use_stress):
        
        energy, forces, virial = atoms.get_efv()

        if use_stress:
            assert virial is not None
        else:
            virial = np.zeros((3,3))

        return energy, forces, virial

    def run(self, atoms_list, use_stress=None):
        assert len(atoms_list) == len(self.hosts)
        if use_stress is None:
            use_stress = False

        nhosts = len(self.hosts)
        msgs = [''] * nhosts

        equeue = list()
        fqueue = list()
        vqueue = list() 

        # status = 1 means sent force
        # status = 2 means sent READY after sent force
        #            which is basically end of one MD
        status = [0] * nhosts  

        while True:

            for i in range(nhosts):
                if sum(status) == 2 * nhosts:
                    # every one finishes their jobs
                    # move on to next step
                    self.istep += 1
                    status = [0] * nhosts

                if status[i] == 2:
                    # this client has done its job
                    # so do not try to receive msg until next step
                    continue
                if i > 0 and status[i-1] < 2:
                    # the previous clients have not finished their jobs
                    # so do not start doing my job
                    continue
                if self.istep % self.paces[i] != 0:
                    # I am not needed at this step
                    # so I finish my job by doing nothing
                    status[i] = 2
                    continue

                try:
                    msgs[i] = self.protocols[i].recvmsg()
                except SocketClosed:
                    msgs[i] = 'EXIT'


                if msgs[i] == 'EXIT':
                    return
                elif msgs[i] == 'STATUS':
                    self.protocols[i].sendmsg(self.states[i])
                    if status[i] > 0:
                        status[i] += 1
                elif msgs[i] == 'POSDATA':
                    assert self.states[i] == 'READY'
                    cell, icell, positions = self.protocols[i].recvposdata()
                    atoms_list[i].set_positions_box(positions, cell)
                    energy, forces, virial = self.calculate(atoms_list[i], use_stress)
                    equeue.append(energy)
                    fqueue.append(forces)
                    vqueue.append(virial)
                    self.states[i] = 'HAVEDATA'
                elif msgs[i] == 'GETFORCE':
                    assert self.states[i] == 'HAVEDATA', self.states[i]
                    self.protocols[i].sendforce(equeue.pop(0), fqueue.pop(0), vqueue.pop(0))
                    self.states[i] = 'READY'
                    status[i] = 1
                else:
                    raise KeyError('Bad message', msgs[i])



class Atoms:
    def __init__(self, efv_scan):
        import copy

        self.efv_scan = efv_scan
        self.init_dict = {'dm0': None}

        self.positions = None
        self.box = None

    def set_positions_box(self, pos, box):
        self.positions = pos
        self.box = box

    def get_efv(self):
        e, f, v, d  = \
            self.efv_scan(self.positions, self.box, self.init_dict)
        self.init_dict = d
        return e, f, v

if __name__ == "__main__":
    from sys import stdout
    import numpy
    import pyscf
    from pyscf import lib, pbc
    from pyscf.pbc import gto as pbcgto
    from pyscf.pbc import dft as pbcdft
    from pyscf.pbc.dft import multigrid
    from pyscf.pbc.grad import rks as rks_grad

    #ABC = numpy.vectorize(float)("9.8752224 9.8752224 9.8752224".split())
    #fp = open("../../spcfw_equil/equil.xyz")
    #fp.readline()
    #fp.readline()
    #xyz = fp.readlines()
    #elements = [l.split()[0] for l in xyz]
    #fp.close()
    elements = \
    ['O','H','H','O','H','H','O','H','H','O','H','H','O','H','H','O','H','H','O','H','H','O','H','H','O','H','H','O','H','H','O','H','H','O','H','H','O','H','H','O','H','H','O','H','H','O','H','H','O','H','H','O','H','H','O','H','H','O','H','H','O','H','H','O','H','H','O','H','H','O','H','H','O','H','H','O','H','H','O','H','H','O','H','H','O','H','H','O','H','H','O','H','H','O','H','H']

    Bohr = 0.52917721092

    def make_atom_str(coords):
        '''
        make PySCF acceptable atom string from coords in Bohr
        '''
        assert len(coords) == len(elements)
        coords = coords * Bohr
        pos2str = lambda pos: " ".join([str(x) for x in pos])
        atom_str = [f"{a} {pos2str(pos)}\n" for a,pos in zip(elements,coords)]
        return atom_str

    def make_mol(coords, box):
        cell = pbcgto.Cell()
        cell.a = box * Bohr
        cell.atom = make_atom_str(coords)
        cell.ke_cutoff = 100  # kinetic energy cutoff in a.u.
        cell.max_memory = 2000 # 2 T
        cell.precision = 1e-6 # integral precision
        cell.pseudo = 'gth-pbe'
        cell.verbose = 1
        cell.basis = {'H': 'gth-dzv', 'O': 'gth-dzv'}
        cell.build()
        return cell

    def make_mol2(coords, box):
        cell = pbcgto.Cell()
        cell.a = box * Bohr
        cell.atom = make_atom_str(coords)
        cell.ke_cutoff = 200  # kinetic energy cutoff in a.u.
        cell.max_memory = 2000 # 2 T
        cell.precision = 1e-6 # integral precision
        cell.pseudo = 'gth-pbe'
        cell.verbose = 1
        cell.basis = {'H': 'gth-dzvp', 'O': 'gth-dzvp'}
        cell.build()
        return cell

    def efv_scan(coords, box, init_dict=None):
        '''
        return energy, force, virial given atom coords[N][3] and box[3][3] in Bohr
        and return init_dict for next step if possible
        kwargs can accept some info from previous MD step \
        to accelerate calculation of this step, such as initial DM
        '''

        cell2 = make_mol2(coords, box)

        df = multigrid.MultiGridFFTDF2(cell2)
        mf = pbcdft.RKS(cell2)
        mf.xc = 'pbe'
        mf.init_guess='atom' # atom guess is fast
        mf.with_df = df
        d3 = None

        if init_dict is not None:
            e = mf.kernel(dm0=init_dict['dm0'])
        else:
            e = mf.kernel()
        f = -rks_grad.Gradients(mf).kernel()
        v = None

        init_dict = {'dm0': mf.make_rdm1()}

        return e, f, v, init_dict

    atoms = Atoms(efv_scan)
    client = SocketClient(unixsocket='driver', log=stdout)
    client.run(atoms)
