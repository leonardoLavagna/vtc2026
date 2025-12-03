import numpy as np
from qiskit import transpile
from qiskit_aer import Aer


backend = Aer.get_backend("aer_simulator")


def curvature(x):
    return 0.0


def expectation_z(qc, n_qubits):
    qc_sv = qc.copy()
    qc_sv.save_statevector()
    compiled = transpile(qc_sv, backend)
    job = backend.run(compiled)
    result = job.result()
    sv = np.array(result.get_statevector(compiled), dtype=np.complex128)
    Z = np.array([[1,0],[0,-1]])
    I = np.eye(2)
    expvals = []
    for target in range(n_qubits):
        Op = 1
        for q in range(n_qubits):
            Op = np.kron(Op, Z if q == target else I)
        ev = np.conj(sv).T @ (Op @ sv)
        expvals.append(np.real(ev))
    return np.array(expvals)
