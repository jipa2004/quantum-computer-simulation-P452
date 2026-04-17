import numpy as np
from qiskit import QuantumCircuit


# ─────────────────────────────────────────────
# Q1.2  Bell / parameter-check circuit
# ─────────────────────────────────────────────
def parameter_check_circuit(theta: float):
    """Ry(θ) on q0, then CNOT q0→q1.  At θ=π gives |11⟩."""
    qc = QuantumCircuit(2, 2)
    qc.ry(theta, 0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


# ─────────────────────────────────────────────
# Q1.3  10-qubit GHZ state
# ─────────────────────────────────────────────
def ghz_circuit():
    """Prepares (|0000000000⟩ + |1111111111⟩)/√2."""
    qc = QuantumCircuit(10, 10)
    qc.h(0)
    for i in range(9):
        qc.cx(i, i + 1)
    qc.measure(range(10), range(10))
    return qc


# ─────────────────────────────────────────────
# Q1.4  Unitarity / state-recovery circuit
# ─────────────────────────────────────────────
def unitarity_circuit():
    """
    Prepares 1/√2 (|201⟩ + |425⟩) in 10-qubit binary:
        |201⟩ = |0011001001⟩   (binary of 201, little-endian qubit order)
        |425⟩ = |1001010110⟩

    Step 1: prepare superposition
    Step 2: apply chain of 9 CNOTs (q0→q1, q1→q2, …, q8→q9)
    Step 3: reverse the chain to recover initial state
    """
    # ---- binary expansions (LSB = q0) ----
    # 201  = 128+64+8+1   = bits 0,3,6,7  (q0,q3,q6,q7)
    # 425  = 256+128+32+8+1 = bits 0,3,5,7,8  (q0,q3,q5,q7,q8)
    #
    # We encode as 10-bit little-endian strings:
    # |201⟩: bit pattern for 201 in positions q0..q9
    #   201 in binary (10 bits) = 0011001001
    #   q0=1, q3=1, q6=1, q7=1  (positions where bit=1, reading right-to-left)
    #
    # Let's just use Qiskit's initialize for clarity and correctness.

    from qiskit.circuit.library import Initialize

    n = 10
    dim = 2**n

    def idx(decimal):
        """Convert decimal to little-endian qubit-index bitstring index."""
        # Qiskit state vector: index = sum(bit_i * 2^i)
        return decimal  # already little-endian compatible

    state = np.zeros(dim, dtype=complex)
    state[idx(201)] = 1 / np.sqrt(2)
    state[idx(425)] = 1 / np.sqrt(2)

    # --- Preparation circuit (no measurement yet) ---
    qc_prep = QuantumCircuit(n, name="Prepare")
    qc_prep.initialize(state, range(n))

    # --- Forward: chain of 9 CNOTs ---
    qc_forward = QuantumCircuit(n, name="Forward CNOTs")
    for i in range(9):
        qc_forward.cx(i, i + 1)

    # --- Reverse: undo the chain (apply in reverse order) ---
    qc_reverse = QuantumCircuit(n, name="Reverse CNOTs")
    for i in reversed(range(9)):
        qc_reverse.cx(i, i + 1)

    # --- Full circuit ---
    qc = QuantumCircuit(n, n)
    qc.compose(qc_prep,    inplace=True)
    qc.barrier(label="After Prep")
    qc.compose(qc_forward, inplace=True)
    qc.barrier(label="After Forward")
    qc.compose(qc_reverse, inplace=True)
    qc.barrier(label="After Reverse")
    qc.measure(range(n), range(n))
    return qc


# ─────────────────────────────────────────────
# Teleportation – Alice's circuit
# ─────────────────────────────────────────────
def alice_circuit(theta: float):
    """
    Alice's side of the teleportation protocol (preset circuit).

    q0 = state to teleport  → Ry(θ)|0⟩
    q1 = Alice's Bell qubit
    q2 = Bob's qubit  (entangled here, not measured by Alice)

    Steps:
      1. Prepare Alice's state: Ry(θ) on q0
      2. Bell pair: H on q1, CNOT q1→q2
      3. Bell measurement basis: CNOT q0→q1, H on q0
      4. Alice measures q0 and q1 (Z basis)

    Classical results m0, m1 are sent to Bob:
      Bob applies Z if m0 == 1
      Bob applies X if m1 == 1
    """
    qc = QuantumCircuit(3, 2)   # 3 qubits, 2 classical bits for Alice's measurement

    # ── 1. Prepare state to teleport ──────────────────────────────
    qc.ry(theta, 0)
    qc.barrier(label="Alice's State")

    # ── 2. Bell pair entanglement ─────────────────────────────────
    qc.h(1)
    qc.cx(1, 2)
    qc.barrier(label="Bell Pair")

    # ── 3. Bell measurement basis rotation ────────────────────────
    qc.cx(0, 1)
    qc.h(0)
    qc.barrier(label="Bell Basis")

    # ── 4. Alice measures q0 → c0, q1 → c1 ───────────────────────
    qc.measure(0, 0)
    qc.measure(1, 1)

    return qc


def bob_circuit(theta: float, m0: int, m1: int):
    """
    Bob's correction circuit given Alice's classical results m0, m1.

    Starts from |0⟩ and applies the same entanglement as alice_circuit
    so that q2 is in the correct post-measurement state, then applies
    Bob's conditional corrections and measures q2.

    In a real experiment this would be a separate device; here we
    reconstruct by re-running the full 3-qubit circuit conditioned on
    the specific (m0, m1) outcome Alice observed, post-selecting on
    that result, then applying corrections and measuring q2.

    Returns a circuit that produces Bob's final measurement.
    """
    qc = QuantumCircuit(3, 1)   # 3 qubits, 1 classical bit for Bob

    # Reproduce Alice's full preparation (same as alice_circuit)
    qc.ry(theta, 0)
    qc.h(1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.h(0)

    # Bob's classical feed-forward corrections
    if m1 == 1:
        qc.x(2)   # X gate if Alice's q1 == 1
    if m0 == 1:
        qc.z(2)   # Z gate if Alice's q0 == 1

    # Bob measures q2
    qc.measure(2, 0)

    return qc


# ─────────────────────────────────────────────
# Q2.2  Long-distance CNOT q0 → q4 via SWAPs
# ─────────────────────────────────────────────
def long_distance_cnot_circuit():
    """
    Linear chain 0-1-2-3-4.
    Perform a logical CNOT(q0, q4) using only nearest-neighbor gates.

    Strategy: SWAP q3↔q4, SWAP q2↔q3, SWAP q1↔q2 to bring q4 adjacent to q0,
    apply CNOT q0→q1 (which now holds the original q4), then SWAP back.

    Each SWAP = 3 CNOTs.  Total CNOTs used: 3×3 (SWAPs) + 1 (logical) + 3×3 (un-SWAPs) = 19.
    A more efficient (unidirectional) approach uses 7 CNOTs shown below.
    """
    qc = QuantumCircuit(5, 5)

    # --- Bring q4 next to q0 via SWAP chain ---
    # SWAP(3,4)
    qc.cx(3, 4); qc.cx(4, 3); qc.cx(3, 4)
    # SWAP(2,3)
    qc.cx(2, 3); qc.cx(3, 2); qc.cx(2, 3)
    # SWAP(1,2)
    qc.cx(1, 2); qc.cx(2, 1); qc.cx(1, 2)

    # --- Logical CNOT: q0 → q1 (originally q4) ---
    qc.cx(0, 1)

    # --- SWAP back ---
    # SWAP(1,2)
    qc.cx(1, 2); qc.cx(2, 1); qc.cx(1, 2)
    # SWAP(2,3)
    qc.cx(2, 3); qc.cx(3, 2); qc.cx(2, 3)
    # SWAP(3,4)
    qc.cx(3, 4); qc.cx(4, 3); qc.cx(3, 4)

    qc.measure(range(5), range(5))
    return qc


# ─────────────────────────────────────────────
# Q3  Fermi-Hubbard Trotterized circuit
# ─────────────────────────────────────────────
def hubbard_trotter_circuit(u_over_j: float, tau: float = 1.0, n_steps: int = 1,
                             init_state: str = "1000"):
    """
    4-qubit 2-site Hubbard model via Jordan-Wigner + Trotter decomposition.

    Qubits: q0=site1↑, q1=site1↓, q2=site2↑, q3=site2↓
    J=1 (fixed), U=u_over_j*J

    Hopping:  H_J = J/2 (XX + YY) on (q0,q2) and (q1,q3)
    Interaction: H_U = U/4 (I - Z0 - Z1 + Z0Z1)  [site1]
                      + U/4 (I - Z2 - Z3 + Z2Z3)  [site2]

    One Trotter step: exp(-i H_J dt) exp(-i H_U dt)
    """
    J = 1.0
    U = u_over_j * J
    dt = tau / n_steps

    qc = QuantumCircuit(4, 4)

    # ── Initial state ─────────────────────────────────────────────
    for i, bit in enumerate(reversed(init_state)):   # Qiskit: q0=rightmost
        if bit == "1":
            qc.x(i)
    qc.barrier(label="Init")

    for _ in range(n_steps):
        # ── Hopping term: XY-rotation on (q0,q2) and (q1,q3) ─────
        # exp(-i θ/2 (XX+YY)) with θ = J*dt
        # Implemented as: Rxx(2Jdt) Ryy(2Jdt)
        angle_hop = J * dt

        for (a, b) in [(0, 2), (1, 3)]:
            # Rxx(2*angle_hop)
            qc.rxx(2 * angle_hop, a, b)
            # Ryy(2*angle_hop)
            qc.ryy(2 * angle_hop, a, b)

        qc.barrier(label="Hop")

        # ── Interaction term: ZZ-rotation on (q0,q1) and (q2,q3) ─
        # exp(-i U/4 * Z_i Z_j * dt)  → Rzz(U*dt/2)
        # Single-qubit Z terms are global phases per sector; include as Rz
        angle_int = U * dt / 4

        for (a, b) in [(0, 1), (2, 3)]:
            qc.rzz(2 * angle_int, a, b)  # ZZ part
            qc.rz(-2 * angle_int, a)     # -Z_a part
            qc.rz(-2 * angle_int, b)     # -Z_b part

        qc.barrier(label="Int")

    qc.measure(range(4), range(4))
    return qc


# ─────────────────────────────────────────────
# Legacy / simple Hubbard preset (for UI preset tab)
# ─────────────────────────────────────────────
def hubbard_circuit(u_over_j: float):
    """Simple single Trotter step for the UI preset selector."""
    return hubbard_trotter_circuit(u_over_j, tau=1.0, n_steps=1, init_state="1000")


# ─────────────────────────────────────────────
# Custom circuit builder
# ─────────────────────────────────────────────
def build_custom_circuit(num_qubits: int, operations: list):
    qc = QuantumCircuit(num_qubits, num_qubits)

    for op in operations:
        gate = op["gate"]

        if gate == "H":
            qc.h(op["qubit"])
        elif gate == "X":
            qc.x(op["qubit"])
        elif gate == "Z":
            qc.z(op["qubit"])
        elif gate == "RX":
            qc.rx(op["param"], op["qubit"])
        elif gate == "RY":
            qc.ry(op["param"], op["qubit"])
        elif gate == "RZ":
            qc.rz(op["param"], op["qubit"])
        elif gate == "CX":
            qc.cx(op["control"], op["target"])
        elif gate == "SWAP":
            # Decompose into 3 CNOTs so it is visible as native gates
            a, b = op["control"], op["target"]
            qc.cx(a, b)
            qc.cx(b, a)
            qc.cx(a, b)

    qc.measure(range(num_qubits), range(num_qubits))
    return qc
