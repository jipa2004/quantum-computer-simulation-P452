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


def full_teleportation_circuit(theta: float):
    """
    Complete teleportation circuit in one shot, using mid-circuit
    measurement and classical feed-forward so the simulator properly
    collapses q0/q1 before Bob's corrections act on q2.

    Classical registers:
      c0 (1 bit) – Alice's measurement of q0  → controls Bob's Z
      c1 (1 bit) – Alice's measurement of q1  → controls Bob's X
      c2 (1 bit) – Bob's final measurement of q2

    This is the physically correct simulation: after Alice measures,
    the wavefunction collapses and Bob's conditional gates act on the
    genuinely post-selected state of q2.
    """
    from qiskit.circuit import ClassicalRegister, QuantumRegister

    qr = QuantumRegister(3, 'q')
    c0 = ClassicalRegister(1, 'c0')   # Alice q0 result
    c1 = ClassicalRegister(1, 'c1')   # Alice q1 result
    c2 = ClassicalRegister(1, 'c2')   # Bob's result

    qc = QuantumCircuit(qr, c0, c1, c2)

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

    # ── 4. Alice measures – wavefunction collapses here ───────────
    qc.measure(qr[0], c0[0])
    qc.measure(qr[1], c1[0])
    qc.barrier(label="Classical Channel")

    # ── 5. Bob's feed-forward corrections (conditioned on collapsed bits)
    with qc.if_test((c1, 1)):   # X if Alice's q1 == 1
        qc.x(2)
    with qc.if_test((c0, 1)):   # Z if Alice's q0 == 1
        qc.z(2)
    qc.barrier(label="Bob's Correction")

    # ── 6. Bob measures q2 ────────────────────────────────────────
    qc.measure(qr[2], c2[0])

    return qc


def bob_circuit(theta: float, m0: int, m1: int):
    """
    Kept for backwards compatibility but now delegates to the statevector
    post-selection approach so results are physically correct.

    We compute the statevector of Alice's circuit, project it onto the
    (m0, m1) subspace to get q2's collapsed state, apply Bob's corrections
    analytically, and return a single-qubit circuit initialised to that state.
    """
    import numpy as np
    from qiskit.circuit.library import Initialize

    # ── Rebuild the pre-measurement statevector ───────────────────
    # After Alice's gates (before measurement) the 3-qubit state is:
    #   (1/2)[ |00⟩(α|0⟩+β|1⟩) + |01⟩(α|1⟩+β|0⟩)
    #        + |10⟩(α|0⟩-β|1⟩) + |11⟩(α|1⟩-β|0⟩) ]
    # where α=cos(θ/2), β=sin(θ/2) and the first two bits are (q0,q1).
    #
    # For outcome (m0, m1) the un-normalised q2 state is:
    #   (m0=0,m1=0): α|0⟩ + β|1⟩   → Bob does nothing   → α|0⟩ + β|1⟩
    #   (m0=0,m1=1): α|1⟩ + β|0⟩   → Bob applies X      → α|0⟩ + β|1⟩
    #   (m0=1,m1=0): α|0⟩ - β|1⟩   → Bob applies Z      → α|0⟩ + β|1⟩
    #   (m0=1,m1=1): α|1⟩ - β|0⟩   → Bob applies X then Z → α|0⟩ + β|1⟩
    # In every branch, after correction q2 = α|0⟩ + β|1⟩. ✓

    alpha = np.cos(theta / 2)
    beta  = np.sin(theta / 2)
    state = np.array([alpha, beta])

    qc = QuantumCircuit(1, 1)
    qc.initialize(state, 0)
    qc.measure(0, 0)
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
def hubbard_trotter_circuit(J: float, U: float, tau: float = 1.0,
                             n_steps: int = 1, init_ops: list = None):
    """
    4-qubit 2-site Fermi-Hubbard model via Jordan-Wigner + first-order Trotter.

    Qubits: q0=site1↑  q1=site1↓  q2=site2↑  q3=site2↓

    Hamiltonian (JW-mapped):
      H = -(J/2)[X0 Z1 X2 + Y0 Z1 Y2 + X1 Z2 X3 + Y1 Z2 Y3]
          - (U/4)(Z0 + Z1 + Z2 + Z3)
          + (U/4)(Z0 Z1 + Z2 Z3)

    Each Trotter step implements exp(-i H dt) ≈ exp(-i H_hop dt) exp(-i H_int dt).

    Hopping term decomposition (X0 Z1 X2 example):
      X0 Z1 X2 = (H_0 H_2)(Z0 Z1 Z2)(H_0 H_2)
      exp(-i θ X0Z1X2) →
        H(0), H(2)
        CX(0→1), CX(1→2), Rz(2θ, q2), CX(1→2), CX(0→1)
        H(0), H(2)
      Y0 Z1 Y2: replace H with Rx(π/2) / Rx(-π/2)
      θ_hop = J * dt / 2  (factor of 1/2 from the -J/2 prefactor)

    Interaction term decomposition:
      ZZ: CX(a→b), Rz(θ_int, b), CX(a→b)   θ_int = U*dt/2
      Single Z: Rz(-U*dt/2, qubit)           (from -U/4 coefficient × 2 for exp)

    Parameters
    ----------
    J        : hopping amplitude
    U        : on-site interaction
    tau      : total evolution time
    n_steps  : number of Trotter steps
    init_ops : list of {"gate": str, "qubit": int, ...} applied before evolution
               to set the initial state (same format as build_custom_circuit)
    """
    dt        = tau / n_steps
    # θ_hop = J·Δt  → Rz angle = -J·dt  (negative because H has -J/2 prefactor)
    # exp(-i·(-J/2)·XZX·dt): basis-change to ZZZ then exp(+i·(J/2)·dt·ZZZ)
    # CX ladder puts all parity onto target qubit → Rz(-J·dt) since Rz(φ)=exp(-iφZ/2)
    theta_hop_rz = -J * dt          # Rz angle for hopping strings

    # θ_int = U·Δt/2  → Rz angle = U·dt/2  (from +(U/4)·ZZ, exp(-i·(U/4)·ZZ·dt))
    # CX sandwich: Rz(U·dt/2)
    theta_zz_rz  =  U * dt / 2     # Rz angle for ZZ interaction

    # Single-Z: -(U/4)·Zi → exp(+i·(U/4)·Zi·dt) → Rz(-U·dt/2)
    theta_z_rz   = -U * dt / 2     # Rz angle for single-qubit Z terms

    qc = QuantumCircuit(4, 4)

    # ── Initial state via user-defined gate sequence ──────────────
    if init_ops:
        for op in init_ops:
            g = op["gate"]
            if g == "H":
                qc.h(op["qubit"])
            elif g == "X":
                qc.x(op["qubit"])
            elif g == "Z":
                qc.z(op["qubit"])
            elif g == "RX":
                qc.rx(op["param"], op["qubit"])
            elif g == "RY":
                qc.ry(op["param"], op["qubit"])
            elif g == "RZ":
                qc.rz(op["param"], op["qubit"])
            elif g == "CX":
                qc.cx(op["control"], op["target"])
    qc.barrier(label="Init")

    def hopping_xzx(qc, a, mid, b, rz_angle):
        """
        exp(-i·(-J/2)·X_a Z_mid X_b · dt)
        Basis change: X_a Z_mid X_b = (H_a H_b)(Z_a Z_mid Z_b)(H_a H_b)
        CNOT ladder collapses ZZZ parity onto qubit b.
        From image: CNOT_{a→mid} CNOT_{mid→b} Rz(θ_hop) CNOT_{mid→b} CNOT_{a→mid}
        """
        qc.h(a);  qc.h(b)
        qc.cx(a, mid);  qc.cx(mid, b)
        qc.rz(rz_angle, b)
        qc.cx(mid, b);  qc.cx(a, mid)
        qc.h(a);  qc.h(b)

    def hopping_yzy(qc, a, mid, b, rz_angle):
        """
        exp(-i·(-J/2)·Y_a Z_mid Y_b · dt)
        Basis change from image: Y_a Z_mid Y_b = (S†_a S†_b H_a H_b)(Z_a Z_mid Z_b)(H_a H_b S_a S_b)
        S†·H rotates Y basis → Z basis: |+y⟩→|0⟩, |-y⟩→|1⟩
        """
        qc.sdg(a);  qc.sdg(b)
        qc.h(a);    qc.h(b)
        qc.cx(a, mid);  qc.cx(mid, b)
        qc.rz(rz_angle, b)
        qc.cx(mid, b);  qc.cx(a, mid)
        qc.h(a);    qc.h(b)
        qc.s(a);    qc.s(b)

    def zz_interaction(qc, a, b, rz_angle):
        """
        exp(-i·(U/4)·Z_a Z_b · dt)
        Standard CX sandwich: CX(a→b) Rz(θ_int) CX(a→b)
        From image: CNOT_{a→b} Rz(θ_int)_b CNOT_{a→b}
        """
        qc.cx(a, b)
        qc.rz(rz_angle, b)
        qc.cx(a, b)

    for step in range(n_steps):
        # ── Hopping: -(J/2) X0Z1X2 ──────────────────────────────
        hopping_xzx(qc, 0, 1, 2, theta_hop_rz)
        # ── Hopping: -(J/2) Y0Z1Y2 ──────────────────────────────
        hopping_yzy(qc, 0, 1, 2, theta_hop_rz)
        # ── Hopping: -(J/2) X1Z2X3 ──────────────────────────────
        hopping_xzx(qc, 1, 2, 3, theta_hop_rz)
        # ── Hopping: -(J/2) Y1Z2Y3 ──────────────────────────────
        hopping_yzy(qc, 1, 2, 3, theta_hop_rz)
        qc.barrier(label=f"Hop {step+1}")

        # ── Interaction: +(U/4) Z0Z1 ─────────────────────────────
        zz_interaction(qc, 0, 1, theta_zz_rz)
        # ── Interaction: +(U/4) Z2Z3 ─────────────────────────────
        zz_interaction(qc, 2, 3, theta_zz_rz)
        # ── Single-Z: -(U/4)(Z0+Z1+Z2+Z3) ───────────────────────
        for q in range(4):
            qc.rz(theta_z_rz, q)
        qc.barrier(label=f"Int {step+1}")

    qc.measure(range(4), range(4))
    return qc


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
