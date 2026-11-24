import numpy as np
from scipy.linalg import logm, inv, sqrtm
from itertools import product
from scipy.optimize import minimize
import warnings


def depolarizing_strength(scores):
    """Convert raw two-qubit overlap scores s=4F to lambda for the depolarizing model."""
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Overlap scores must be finite.")
    return (1 - scores / 4) / (1 - 1 / 16)


def tomography_matrices():
    """Return the recorded six-state preparation and three-axis measurement design."""
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.diag([1, -1]).astype(complex)
    axes = (Z, X, Y)
    states = [(I + sign * axis) / 2 for axis in axes for sign in (1, -1)]
    inputs = [np.kron(states[j], states[i]) for i, j in product(range(6), repeat=2)]
    outputs = [np.kron((I + (-1)**((m >> 1) & 1) * axes[l]) / 2,
                       (I + (-1)**(m & 1) * axes[k]) / 2)
               for k, l in product(range(3), repeat=2) for m in range(4)]
    return inputs, outputs


def marginal_counts(probs, qubits):
    """Sum full-register counts over the selected classical bit positions."""
    qubits = sorted(qubits)
    marginals = []
    for prob in probs:
        counts = {}
        for key, value in prob.items():
            bits = key.replace(" ", "")
            outcome = sum(int(bits[-1-qubit]) << i for i, qubit in enumerate(qubits))
            label = format(outcome, f"0{len(qubits)}b")
            counts[label] = counts.get(label, 0) + value
        marginals.append(counts)
    return marginals


class Hamiltonian_Learning:

    def __init__(self, num_qubits):
        self.num_qubits = num_qubits

        self._states, self._measures = tomography_matrices()
        self._lambda = None
        self.fit_diagnostics = []

    def groups_qubits_zeno( self ):
        num_qubits = self.num_qubits
        qubits_zeno_1 = np.arange(2,num_qubits,3)
        qubits_zeno_2 = np.arange(0,num_qubits,3)
        qubits_zeno_3 = np.arange(1,num_qubits,3)
        return [qubits_zeno_1,qubits_zeno_2,qubits_zeno_3]

    def groups_qubits_tomo(self):

        num_qubits = self.num_qubits
        qubits_zeno_1, qubits_zeno_2, qubits_zeno_3 = self.groups_qubits_zeno()

        qubits_tomo_1 = list(range(num_qubits))
        for qubit in qubits_zeno_1:
            qubits_tomo_1.remove(qubit)
        qubits_tomo_1 = np.array(qubits_tomo_1)[:2*(len(qubits_tomo_1)//2)].reshape(-1,2)

        qubits_tomo_2 = list(range(num_qubits))
        for qubit in qubits_zeno_2:
            qubits_tomo_2.remove(qubit)
        qubits_tomo_2 = np.array(qubits_tomo_2[:2*(len(qubits_tomo_2)//2)]).reshape(-1,2)

        qubits_tomo_3 = list(range(num_qubits))
        for qubit in qubits_zeno_3:
            qubits_tomo_3.remove(qubit)
        qubits_tomo_3 = np.array(qubits_tomo_3[1:2*(len(qubits_tomo_3)//2)+1]).reshape(-1,2)

        qubits_tomo = [qubits_tomo_1, qubits_tomo_2, qubits_tomo_3 ]

        return qubits_tomo

    def trotter_evol_with_kicks( self, trotter_step, number_steps ):

        from qiskit import QuantumCircuit

        num_qubits = self.num_qubits
        qubits_zeno =  self.groups_qubits_zeno()

        # first group
        qc_group_1 = QuantumCircuit( num_qubits, num_qubits-len(qubits_zeno[0]) )
        for _ in range(number_steps):
            qc_group_1.z(np.arange(2,num_qubits,3))
            qc_group_1.compose( trotter_step, qubits=range(num_qubits), inplace=True )

        # second group
        qc_group_2 = QuantumCircuit( num_qubits, num_qubits-len(qubits_zeno[1]) )
        for _ in range(number_steps):
            qc_group_2.z(qubits_zeno[1])
            qc_group_2.compose( trotter_step, qubits=range(num_qubits), inplace=True )

        # third group
        qc_group_3 = QuantumCircuit( num_qubits, num_qubits-len(qubits_zeno[2]) )
        for _ in range(number_steps):
            qc_group_3.z(qubits_zeno[2])
            qc_group_3.compose( trotter_step, qubits=range(num_qubits), inplace=True )

        return [ qc_group_1, qc_group_2, qc_group_3 ]


    def circuits_tomo( self, trotter_step, number_steps  ):

        from qiskit_experiments.library.tomography.basis.pauli_basis import PauliMeasurementBasis, Pauli6PreparationBasis

        inputs  = Pauli6PreparationBasis()
        outputs = PauliMeasurementBasis()

        qubit_groups = self.groups_qubits_tomo()
        qc_groups    = self.trotter_evol_with_kicks( trotter_step, number_steps )

        qc_tomo_group = []
        for n, qubits in enumerate(qubit_groups):
            for i, j in product(range(6),range(6)):
                for k, l in product(range(3),range(3)):
                    qc_temp = qc_groups[n].copy()
                    for m, qubit in enumerate(qubits):
                        # print( qubit )
                        qc_temp.compose(inputs.circuit([i,j]), qubits=qubit, front=True, inplace=True )
                        qc_temp.compose(outputs.circuit([k,l]), qubits=qubit, clbits=[2*m,2*m+1], inplace=True )
                    qc_tomo_group.append( qc_temp.decompose() )

        return qc_tomo_group

    def Process2Choi(self, Process):
        dim = int(np.sqrt(Process[0,:].size))
        Process = np.transpose(Process.reshape([dim,dim,dim,dim]),
                                [0,2,1,3]).reshape([dim**2,dim**2])
        return Process

    def complex2unitary( self, Z ):

        if Z.ndim == 3:
            for k, z in enumerate(Z):
                Z[k] = z @ inv( sqrtm( z.T.conj()@z ) )
        else:
            Z = Z @ inv( sqrtm( Z.T.conj()@Z ) )

        return Z

    def linear_tomo( self, probs, qubits, unitary=True, opt=True, init=None ):

        states = self._states
        measures = self._measures
        probs_marginal = marginal_counts(probs, qubits)

        P = np.zeros( (9,4,36) )
        temp = 0
        for i in range( 36 ):
            for j in range(9):
                prob = probs_marginal[temp]
                temp = temp + 1
                for k in prob:
                    P[j,int(k,2),i] = prob[k]
        P = P.reshape(36,36)
        P = 9 * P / np.sum(P,axis=0)

        R = np.array(states).reshape(36,16).T
        M = np.array(measures).reshape(36,16).conj()

        # Linear tomography
        process = np.linalg.pinv(M)@P@np.linalg.pinv(R.T.conj()).T.conj()

        Choi = self.Process2Choi( process )

        if unitary:
            if opt:
                def fun( U ):
                    U = U[:16] + 1j*U[16:]
                    U = self.complex2unitary( U.reshape(4,4) )
                    Psi = np.kron( U, np.eye(4) )@( np.eye(4).reshape(-1) )
                    return - np.vdot( Psi, Choi@Psi ).real

                if init is None:
                    init = np.random.randn(32)
                else:
                    init = np.hstack( [ init.real.reshape(-1), init.imag.reshape(-1)])

                results = minimize( fun, init.reshape(-1) )
                self.fit_diagnostics.append({
                    "success": bool(results.success), "status": int(results.status),
                    "message": str(results.message), "objective": float(results.fun),
                })
                if not np.isfinite(results.x).all() or not np.isfinite(results.fun):
                    raise ValueError("The unitary fit returned non-finite values.")
                if not results.success:
                    warnings.warn("Unitary fit did not converge; inspect fit_diagnostics.", RuntimeWarning)
                U = results.x
                U = U[:16] + 1j*U[16:]
                U = self.complex2unitary( U.reshape(4,4) )
                Uvec = U.reshape(-1)
                Uvec = Uvec / np.linalg.norm( Uvec )
                lambdaa = np.vdot( Uvec, Choi@Uvec ).real
            else:
                eigvals, eigvecs = np.linalg.eigh( Choi )
                # print( eigvals[-1], eigvals[-2], eigvals[-3], eigvals[0])
                lambdaa = eigvals[-1]
                U = np.sqrt(eigvals[-1]) * eigvecs[:,-1].reshape(4,4)
                U = self.complex2unitary( U )

            if self._lambda is None:
                    self._lambda = [ lambdaa ]
            else:
                self._lambda.append( lambdaa )
            return U
        else:
            return Choi

    def noise(self, probs=None, opt=True, init=None):
        """Return raw overlap scores s=4F, not depolarizing strengths."""
        if self._lambda is None:
            if probs is None:
                raise ValueError("Provide tomography data or reconstruct the channel first.")
            self.unitaries_from_tomo(probs=probs, opt=opt, init=init)

        return self._lambda

    def unitaries_from_tomo( self, probs, opt=True, init=None ):

        self._lambda = None
        self.fit_diagnostics = []

        probs_1 = probs[:36*9]
        probs_2 = probs[36*9:36*9*2]
        probs_3 = probs[36*9*2:]

        Us = []
        qubits_tomo = self.groups_qubits_tomo()

        for j,_ in enumerate( qubits_tomo[0] ):
            U = self.linear_tomo( probs_1, [2*j,2*j+1], opt=opt, init=init )
            Us.append( U )
        for j,_ in enumerate( qubits_tomo[1] ):
            U = self.linear_tomo( probs_2, [2*j,2*j+1], opt=opt, init=init )
            Us.append( U )
        for j,_ in enumerate( qubits_tomo[2] ):
            U = self.linear_tomo( probs_3, [2*j,2*j+1], opt=opt, init=init )
            Us.append( U )

        return Us

    def choi_from_tomo( self, probs, opt=True, init=None ):

        probs_1 = probs[:36*9]
        probs_2 = probs[36*9:36*9*2]
        probs_3 = probs[36*9*2:]

        Us = []
        qubits_tomo = self.groups_qubits_tomo()

        for j,_ in enumerate( qubits_tomo[0] ):
            U = self.linear_tomo( probs_1, [2*j,2*j+1], opt=opt, init=init, unitary=False )
            Us.append( U )
        for j,_ in enumerate( qubits_tomo[1] ):
            U = self.linear_tomo( probs_2, [2*j,2*j+1], opt=opt, init=init, unitary=False  )
            Us.append( U )
        for j,_ in enumerate( qubits_tomo[2] ):
            U = self.linear_tomo( probs_3, [2*j,2*j+1], opt=opt, init=init, unitary=False  )
            Us.append( U )

        return Us

    def hamiltonian_from_tomo( self, probs, opt=True, init=None ):

        Us  = self.unitaries_from_tomo( probs, opt=opt, init=init )

        Hs = []
        for U in Us:
            trace = np.trace(U)
            if abs(trace) <= len(U) * np.finfo(float).eps:
                raise ValueError("Unitary trace is too small to fix the global phase.")
            U = U * np.exp(-1j*np.angle(trace))
            Hs.append( 1j*logm( U ) )

        return Hs



def Outer2Kron( A, Dims ):
    # From vec(A) outer vec(B) to A kron B
    N   = len(Dims)
    Dim = A.shape
    A   = np.transpose( A.reshape(2*Dims), np.array([range(N),range(N,2*N) ]).T.flatten() ).flatten()
    return A.reshape(Dim)

def Kron2Outer( A, Dims ):
    # From A kron B to vec(A) outer vec(B)
    N   = len(Dims)
    Dim = A.shape
    A   = np.transpose( A.reshape( np.kron(np.array([1,1]),Dims) ), np.array([range(0,2*N,2),range(1,2*N,2)]).flatten() ).flatten()
    return A.reshape(Dim)

def LocalProduct( Psi, Operators , Dims=[] ):
    """
    Calculate the product (A1xA2x...xAn)|psi>
    """
    sz = Psi
    if not Dims:
        Dims = [ Operators[k].shape[-1] for k in range( len(Operators) ) ]
    N = len(Dims)
    for k in range(N):
        Psi  = (( Operators[k]@Psi.reshape(Dims[k],-1) ).T ).flatten()
    return Psi

def InnerProductMatrices( X, B, Vectorized = False ):
    """
    Calculate the inner product tr( X [B1xB2x...xBn])
    """
    X = np.array(X)

    if isinstance(B, list):
        B = B.copy()
        nsys = len(B)
        nops = []
        Dims = []
        if Vectorized == False :
            for j in range(nsys):
                B[j] = np.array(B[j])
                if B[j].ndim == 2 :
                    B[j] = np.array([B[j]])
                nops.append( B[j].shape[0] )
                Dims.append( B[j].shape[1] )
                B[j] = B[j].reshape(nops[j],Dims[j]**2)
        elif Vectorized == True :
            for j in range(nsys):
                nops.append( B[j].shape[0] )
                Dims.append( int(np.sqrt(B[j].shape[1])) )

        if X.ndim == 2 :
            TrXB = LocalProduct( Outer2Kron( X.flatten(), Dims ), B )
        elif X.ndim == 3 :
            TrXB = []
            for j in range( X.shape[0] ):
                TrXB.append( LocalProduct( Outer2Kron( X[j].flatten(), Dims ), B ) )
        elif X.ndim == 1:
            TrXB = LocalProduct( Outer2Kron( X, Dims ), B )

        return np.array( TrXB ).reshape(nops)

    elif isinstance(B, np.ndarray):

        if B.ndim == 2 and Vectorized == False :
            return np.trace( X @ B )

        elif B.ndim == 4 :
            nsys = B.shape[0]
            nops = nsys*[ B[0].shape[0] ]
            Dims = nsys*[ B[0].shape[1] ]
            B = B.reshape(nsys,nops[0],Dims[0]**2)

        elif B.ndim == 3 :
            if Vectorized == False :
                nsys = 1
                nops = B.shape[0]
                Dims = [ B.shape[1] ]
                B = B.reshape(nsys,nops,Dims[0]**2)
            if Vectorized == True :
                nsys = B.shape[0]
                nops = nsys*[ B[0].shape[0] ]
                Dims = nsys*[ int(np.sqrt(B[0].shape[1])) ]
        if X.ndim == 2 :
            TrXB = LocalProduct( Outer2Kron( X.flatten(), Dims ), B )
        elif X.ndim == 3 :
            TrXB = []
            for j in range( X.shape[0] ):
                TrXB.append( LocalProduct( Outer2Kron( X[j].flatten(), Dims ), B ) )
        elif X.ndim == 1:
            TrXB = LocalProduct( Outer2Kron( X, Dims ), B )

        return np.array( TrXB ).reshape(nops)

def Pauli_Transfer_Matrix( H, num_qubits, normalize=True, real=True ):
    """
    Calcula the matrix M_i1i2...iN =Tr( H [P_i1xP_i2x...xP_iN] )/2**num_qubits,
    with P_i  the Pauli matrices. The coefficiets M_i1i2...iN permit to write H as

    H = sum_i1i2...iN  M_i1i2...iN P_i1xP_i2x...xP_iN

    The setting normalize=False remove the normalization 1/2**num_qubits of M.
    """

    B = np.zeros((4,2,2), dtype=complex )
    B[0] = np.eye(2)
    B[1] = np.array([0,1,1,0]).reshape(2,2)
    B[2] = np.array([0,-1j,1j,0]).reshape(2,2)
    B[3] = np.diag([1,-1])

    PTM = InnerProductMatrices( H, np.repeat( [B], num_qubits, axis=0 ) )

    if real:
        PTM = PTM.real

    if normalize:
        PTM = PTM / 2**num_qubits

    return PTM
