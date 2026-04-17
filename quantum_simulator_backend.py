from qiskit import transpile, QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np

# Shot-based simulator
backend = AerSimulator(method="automatic")

# Statevector simulator
sv_backend = AerSimulator(method="statevector")


def run_and_get_histogram(circuit: QuantumCircuit, shots: int = 1024) -> dict:
    if circuit.num_qubits > 10:
        raise ValueError("Circuit exceeds the 10-qubit limit for this backend.")

    if not circuit.cregs:
        raise ValueError("Circuit must include classical registers for measurement.")

    compiled = transpile(circuit, backend)
    result = backend.run(compiled, shots=shots).result()
    return result.get_counts()


def run_single_shot(circuit: QuantumCircuit) -> str:
    """
    Run circuit for exactly 1 shot and return the bitstring result.
    Useful for simulating individual runs of a protocol.
    """
    compiled = transpile(circuit, backend)
    result = backend.run(compiled, shots=1).result()
    counts = result.get_counts()
    return list(counts.keys())[0]


def get_statevector(circuit: QuantumCircuit) -> np.ndarray:
    qc_no_meas = circuit.remove_final_measurements(inplace=False)
    qc_no_meas.save_statevector()

    compiled = transpile(qc_no_meas, sv_backend)
    result = sv_backend.run(compiled).result()

    return np.array(result.get_statevector())
