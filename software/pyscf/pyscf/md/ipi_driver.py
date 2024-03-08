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

        energy, forces = atoms.get_energy_grad()
        if use_stress:
            stress = atoms.get_stress(voigt=False)
            virial = -atoms.get_volume() * stress
        else:
            virial = np.zeros((3, 3))
        return energy, -forces, virial

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
                    # TODO icell ????
                    atoms.cell[:] = cell
                    atoms.set_positions(positions)

                    energy, forces, virial = self.calculate(atoms, use_stress)

                    self.state = 'HAVEDATA'
                    # @@@@@@@@@@@@@@@
                    self.log("positions", positions)
                    yield
                elif msg == 'GETFORCE':
                    assert self.state == 'HAVEDATA', self.state
                    self.protocol.sendforce(energy, forces, virial)
                    self.state = 'READY'
                    # @@@@@@@@@@@@@@@
                    self.log("energy force", energy, forces)
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


class Atoms:
    # TODO update both positions when doing dual basis ?
    def __init__(self, method, kwargs, fakecell=None):
        import copy

        self.mol = method.mol.copy()
        self.method = method
        self.dm = None
        if 'mol2' in kwargs:
            self.mol2 = kwargs.pop('mol2')
        else:
            self.mol2 = None
        self.kwargs = kwargs

        self.fakecell = fakecell
        if self.fakecell is not None:
            self.fakecell = np.array(fakecell)

    @property
    def cell(self):
        if hasattr(self.method, "cell") and hasattr(self.method.cell, "a"):
            return self.method.cell.a
        else:
            assert self.fakecell is not None
            return self.fakecell

    def set_positions(self, positions):
        self.mol.set_geom_(positions, unit='Bohr')
        self.mol.build()
        if self.mol2 is not None:
            self.mol2.set_geom_(positions, unit='Bohr')
            self.mol2.build()

    def get_energy_grad(self):
        if self.mol2 is None:
            self.method.__init__(self.mol, **self.kwargs)
        else:
            self.method.__init__(self.mol, self.mol2, **self.kwargs)
        etot = self.method.kernel(dm0=self.dm)
        if self.mol2 is None:
            self.dm = self.method.make_rdm1()
        elif hasattr(self.method, 'dm_small') and \
             self.method.dm_small is not None:
            self.dm = self.method.dm_small
        return etot, self.method.nuc_grad_method().kernel()

    def get_stress(self):
        raise NotImplementedError

    def get_volume(self):
        raise NotImplementedError


if __name__ == "__main__":
    from sys import stdout
    from pyscf import gto, dft

    fp = open("init.xyz")
    fp.readline()
    fp.readline()
    mol = gto.M(atom=fp.readlines(), basis='sto-3g')
    fp.close()
    kwargs = {'xc': 'blyp'}
    mf = dft.rks.RKS(mol, **kwargs)
    atoms = Atoms(mf, kwargs, fakecell=np.zeros((3, 3)))

    client = SocketClient(unixsocket='driver', log=stdout)
    client.run(atoms)
