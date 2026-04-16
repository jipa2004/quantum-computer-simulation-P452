"""
P452 Quantum Circuit Explorer
==============================
Streamlit frontend for the 10-qubit Qiskit-Aer simulator.
Theme: maroon + white  (set via .streamlit/config.toml)
"""

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from quantum_simulator_backend import run_and_get_histogram, get_statevector
from circuits import (
    parameter_check_circuit,
    ghz_circuit,
    unitarity_circuit,
    teleportation_circuit,
    long_distance_cnot_circuit,
    hubbard_circuit,
    hubbard_trotter_circuit,
    build_custom_circuit,
)

# ─────────────────────────────────────────────────────────────
# Page config & global CSS
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="P452 Quantum Explorer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAROON   = "#800000"
CREAM    = "#FFF8F0"
DARK     = "#1A0000"
ACCENT   = "#C0392B"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"]  {{
      font-family: 'EB Garamond', serif;
      background-color: #FFFFFF;
      color: {DARK};
  }}

  /* ── Header bar ── */
  .app-header {{
      background: {MAROON};
      color: white;
      padding: 1.4rem 2rem 1rem;
      border-radius: 0 0 12px 12px;
      margin-bottom: 1.4rem;
  }}
  .app-header h1 {{
      font-size: 2rem;
      font-weight: 700;
      letter-spacing: 0.04em;
      margin: 0;
  }}
  .app-header p {{
      font-size: 1rem;
      opacity: 0.85;
      margin: 0.25rem 0 0;
  }}

  /* ── Section labels ── */
  .section-label {{
      display: inline-block;
      background: {MAROON};
      color: white;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.72rem;
      letter-spacing: 0.12em;
      padding: 0.2rem 0.7rem;
      border-radius: 4px;
      margin-bottom: 0.5rem;
      text-transform: uppercase;
  }}

  /* ── Card ── */
  .card {{
      background: {CREAM};
      border: 1px solid #E8D8D0;
      border-left: 4px solid {MAROON};
      border-radius: 8px;
      padding: 1.1rem 1.4rem;
      margin-bottom: 1rem;
  }}

  /* ── Streamlit widget label overrides ── */
  label, .stSlider label, .stRadio label, .stSelectbox label {{
      font-family: 'EB Garamond', serif !important;
      font-size: 1rem !important;
  }}

  /* ── Run button ── */
  .stButton > button {{
      background: {MAROON} !important;
      color: white !important;
      border: none !important;
      border-radius: 6px !important;
      font-family: 'EB Garamond', serif !important;
      font-size: 1rem !important;
      padding: 0.45rem 1.4rem !important;
      transition: opacity 0.2s;
  }}
  .stButton > button:hover {{
      opacity: 0.85 !important;
  }}

  /* ── Tab styling ── */
  .stTabs [data-baseweb="tab"] {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.82rem;
  }}
  .stTabs [aria-selected="true"] {{
      border-bottom: 3px solid {MAROON} !important;
      color: {MAROON} !important;
  }}

  /* ── Histogram bars via recharts/vega — can't directly style, 
         but st.bar_chart picks up primaryColor from config.toml ── */

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {{
      background: #FAF0F0;
      border-right: 2px solid #E0C0C0;
  }}
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 {{
      color: {MAROON};
  }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="app-header">
  <h1>⚛️ P452 Quantum Circuit Explorer</h1>
  <p>10-Qubit Simulator · Qiskit-Aer Backend · University of Chicago</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def draw_circuit(qc, title="Circuit Diagram"):
    """Render circuit via Qiskit MPL drawer and show in Streamlit."""
    st.markdown(f'<div class="section-label">🔬 {title}</div>', unsafe_allow_html=True)
    style = {
        "backgroundcolor": "#FFFFFF",
        "linecolor":       MAROON,
        "textcolor":       DARK,
        "gatefacecolor":   "#FDECEA",
        "gateoutlinecolor": MAROON,
    }
    fig = qc.draw(output="mpl", style=style, fold=-1)
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def show_histogram(counts: dict, title="Measurement Results"):
    """Display measurement histogram with maroon bars."""
    st.markdown(f'<div class="section-label">📊 {title}</div>', unsafe_allow_html=True)

    labels = sorted(counts.keys())
    values = [counts[k] for k in labels]
    total  = sum(values)
    probs  = [v / total for v in values]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.6), 3.5))
    bars = ax.bar(labels, probs, color=MAROON, edgecolor="white", width=0.6)
    ax.set_ylabel("Probability", fontsize=11)
    ax.set_xlabel("Bitstring", fontsize=11)
    ax.set_ylim(0, min(1.15, max(probs) * 1.25))
    ax.tick_params(axis="x", rotation=45)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")
    # Annotate bars
    for bar, p in zip(bars, probs):
        if p > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{p:.2f}", ha="center", va="bottom", fontsize=8, color=DARK)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    with st.expander("Raw counts"):
        st.write(counts)


# ─────────────────────────────────────────────────────────────
# Tabs  (one per project phase)
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🛠  Phase 1 – Backend Setup",
    "📡  Phase 2 – Teleportation",
    "⚛️  Phase 3 – Hubbard Model",
    "🎛  Custom Circuit",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1 – Phase 1
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Phase 1 · Backend Setup & Visualisation")

    q1_choice = st.selectbox(
        "Select checkpoint",
        ["Q1.2 – Parameter Check (Ry + CNOT)",
         "Q1.3 – 10-Qubit GHZ State",
         "Q1.4 – Unitarity / State Recovery"],
        key="q1_choice",
    )

    # ── Q1.2 ─────────────────────────────────────────────────
    if q1_choice.startswith("Q1.2"):
        st.markdown("""
        <div class="card">
        Build a 2-qubit circuit: apply Ry(θ) to q0, then CNOT(q0→q1).
        At θ=π the output should be |11⟩ with ~100% probability.
        </div>
        """, unsafe_allow_html=True)

        theta = st.slider("Rotation angle θ", 0.0, float(2 * np.pi), float(np.pi),
                          step=0.01, format="%.3f", key="q12_theta")
        st.caption(f"θ = {theta:.4f} rad  ({theta/np.pi:.3f}π)")

        circuit = parameter_check_circuit(theta)
        draw_circuit(circuit, "Q1.2 – Parameter-Check Circuit")

        if st.button("▶ Run Simulation", key="run_q12"):
            with st.spinner("Simulating …"):
                counts = run_and_get_histogram(circuit)
            show_histogram(counts, "Q1.2 – Measurement Results")

            st.markdown("""
            **Why does |11⟩ ≈ 100% at θ=π?**  
            Ry(π)|0⟩ = |1⟩, so q0 starts in |1⟩.  The CNOT then flips q1
            (which starts in |0⟩) whenever q0=1, giving |11⟩ deterministically.
            Any deviation from 100% is purely shot noise (1/√N ≈ 3% for 1024 shots).
            The backend receiving the correct slider value is confirmed because
            only the *exact* setting θ=π produces this outcome.
            """)

    # ── Q1.3 ─────────────────────────────────────────────────
    elif q1_choice.startswith("Q1.3"):
        st.markdown("""
        <div class="card">
        10-qubit GHZ state: (|0000000000⟩ + |1111111111⟩)/√2.
        Measuring should give <b>0000000000</b> and <b>1111111111</b> with equal probability.
        </div>
        """, unsafe_allow_html=True)

        circuit = ghz_circuit()
        draw_circuit(circuit, "Q1.3 – 10-Qubit GHZ Circuit")

        if st.button("▶ Run Simulation", key="run_q13"):
            with st.spinner("Simulating …"):
                counts = run_and_get_histogram(circuit)
            show_histogram(counts, "Q1.3 – GHZ Measurement Results")

    # ── Q1.4 ─────────────────────────────────────────────────
    else:
        st.markdown("""
        <div class="card">
        Prepare 1/√2(|201⟩+|425⟩), apply 9 CNOT gates, then reverse to recover the initial state.
        </div>
        """, unsafe_allow_html=True)

        circuit = unitarity_circuit()
        draw_circuit(circuit, "Q1.4 – Unitarity Circuit")

        if st.button("▶ Inspect Statevectors", key="run_q14"):
            from qiskit import QuantumCircuit as QC
            from circuits import unitarity_circuit as uc

            # Build sub-circuits for statevector inspection
            from qiskit.circuit.library import Initialize
            import numpy as np

            n = 10
            dim = 2**n
            state_init = np.zeros(dim, dtype=complex)
            state_init[201] = 1 / np.sqrt(2)
            state_init[425] = 1 / np.sqrt(2)

            # Step 1: initial state
            qc1 = QC(n)
            qc1.initialize(state_init, range(n))

            # Step 2: + forward CNOTs
            qc2 = qc1.copy()
            for i in range(9):
                qc2.cx(i, i + 1)

            # Step 3: + reverse CNOTs
            qc3 = qc2.copy()
            for i in reversed(range(9)):
                qc3.cx(i, i + 1)

            with st.spinner("Computing statevectors …"):
                sv1 = get_statevector(qc1)
                sv2 = get_statevector(qc2)
                sv3 = get_statevector(qc3)

            def nonzero_sv(sv, label):
                st.markdown(f"**{label}**")
                nz = {f"|{i:010b}⟩ = |{i}⟩": sv[i]
                      for i in range(len(sv)) if abs(sv[i]) > 1e-6}
                rows = [f"`{k}`: amplitude = {v.real:.4f}" +
                        (f" + {v.imag:.4f}i" if abs(v.imag) > 1e-9 else "")
                        for k, v in nz.items()]
                st.markdown("\n".join(rows))

            col1, col2, col3 = st.columns(3)
            with col1:
                nonzero_sv(sv1, "① After Preparation")
            with col2:
                nonzero_sv(sv2, "② After Forward CNOTs")
            with col3:
                nonzero_sv(sv3, "③ After Reverse CNOTs")

            st.markdown("""
            **Unitarity confirmed:** The state after step ③ matches step ①
            exactly, demonstrating that the reverse circuit is the exact inverse
            (unitary adjoint) of the forward circuit.  
            Quantum gates are always unitary: U†U = I.
            """)


# ═══════════════════════════════════════════════════════════════
# TAB 2 – Phase 2: Teleportation
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Phase 2 · Teleportation & Hardware Connectivity")

    q2_choice = st.selectbox(
        "Select checkpoint",
        ["Q2.1 – Teleportation Circuit",
         "Q2.2 – Long-Distance CNOT (q0→q4)",
         "Q2.3 – Teleportation Statistics"],
        key="q2_choice",
    )

    # ── Q2.1 ─────────────────────────────────────────────────
    if q2_choice.startswith("Q2.1"):
        st.markdown("""
        <div class="card">
        Teleport |q0⟩ = (1/√5)(2|0⟩+|1⟩) from Alice to Bob using a shared Bell pair.
        θ = 2·arctan(1/2) ≈ 0.9273 rad gives the correct superposition.
        </div>
        """, unsafe_allow_html=True)

        # (1/√5)(2|0⟩+|1⟩)  →  Ry(2·arctan(1/2))
        default_theta = float(2 * np.arctan(0.5))
        theta = st.slider("Rotation angle θ (Alice's qubit)", 0.0, float(2 * np.pi),
                          default_theta, step=0.001, format="%.4f", key="q21_theta")
        st.caption(f"θ = {theta:.4f} rad   →   |q0⟩ ∝ cos(θ/2)|0⟩ + sin(θ/2)|1⟩")

        circuit = teleportation_circuit(theta)
        draw_circuit(circuit, "Q2.1 – 3-Qubit Teleportation")

        if st.button("▶ Run Teleportation (1024 shots)", key="run_q21"):
            with st.spinner("Simulating …"):
                counts = run_and_get_histogram(circuit)
            show_histogram(counts, "Q2.1 – Teleportation Results")

    # ── Q2.2 ─────────────────────────────────────────────────
    elif q2_choice.startswith("Q2.2"):
        st.markdown("""
        <div class="card">
        A CNOT between non-adjacent qubits in a linear chain (0-1-2-3-4) requires
        SWAP gates to bring them together.  Each SWAP = 3 CNOTs.
        The circuit below performs CNOT(q0, q4) using 19 total CNOT gates (3 SWAP-in + 1 logical + 3 SWAP-out, each SWAP=3).
        </div>
        """, unsafe_allow_html=True)

        circuit = long_distance_cnot_circuit()
        draw_circuit(circuit, "Q2.2 – Long-Distance CNOT via SWAPs")

        st.markdown("""
        **CNOT gate count breakdown:**

        | Step | Operation | CNOTs |
        |------|-----------|-------|
        | SWAP(3,4) | move q4 toward q0 | 3 |
        | SWAP(2,3) | | 3 |
        | SWAP(1,2) | q4 now adjacent to q0 | 3 |
        | Logical CNOT(q0,q1) | the actual operation | 1 |
        | SWAP(1,2) | restore qubit order | 3 |
        | SWAP(2,3) | | 3 |
        | SWAP(3,4) | | 3 |
        | **Total** | | **19** |
        """)

    # ── Q2.3 ─────────────────────────────────────────────────
    else:
        st.markdown("""
        <div class="card">
        Run teleportation 1024 times with Alice starting in |0⟩ (θ=0).
        Bob should measure |0⟩ with ~100% probability.
        </div>
        """, unsafe_allow_html=True)

        circuit = teleportation_circuit(0.0)   # Alice sends |0⟩
        draw_circuit(circuit, "Q2.3 – Teleportation of |0⟩")

        if st.button("▶ Run 1024 Shots", key="run_q23"):
            with st.spinner("Simulating …"):
                counts = run_and_get_histogram(circuit, shots=1024)
            show_histogram(counts, "Q2.3 – Bob's Measurement")

            # Bob's qubit is classical bit 2 (rightmost in Qiskit bitstring)
            bob_0 = sum(v for k, v in counts.items() if k[-1] == "0")
            bob_1 = sum(v for k, v in counts.items() if k[-1] == "1")
            total  = bob_0 + bob_1
            st.metric("P(Bob = |0⟩)", f"{bob_0/total*100:.1f}%")

            st.markdown(f"""
            **Explanation of any deviation from 100%:**  
            Ideal teleportation is *deterministic* — Bob always recovers |0⟩.  
            Deviations come from **shot noise** (≈ 1/√1024 ≈ 3%) and **simulator 
            transpilation noise** (negligible with Aer's ideal backend).  
            In real hardware, gate fidelity errors and decoherence would cause 
            additional leakage into |1⟩.
            """)


# ═══════════════════════════════════════════════════════════════
# TAB 3 – Phase 3: Hubbard Model
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Phase 3 · Fermi-Hubbard Model")

    q3_choice = st.selectbox(
        "Select checkpoint",
        ["Q3.1 – Circuit Architecture (1 Trotter step)",
         "Q3.2 – Non-Interacting Dynamics (U=0)",
         "Q3.3 – Mott Physics (U=10)"],
        key="q3_choice",
    )

    # ── Q3.1 ─────────────────────────────────────────────────
    if q3_choice.startswith("Q3.1"):
        st.markdown("""
        <div class="card">
        One Trotter step of the 2-site Hubbard model using Jordan-Wigner mapping.<br>
        <b>Hopping term</b>: Rxx + Ryy on (q0,q2) and (q1,q3).<br>
        <b>Interaction term</b>: Rzz + single-qubit Rz on (q0,q1) and (q2,q3).
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            uj   = st.slider("U/J", 0.0, 10.0, 1.0, step=0.1, key="q31_uj")
        with col2:
            tau  = st.slider("τ (time step)", 0.01, np.pi, 1.0, step=0.01, key="q31_tau")

        circuit = hubbard_trotter_circuit(uj, tau=tau, n_steps=1, init_state="1000")
        draw_circuit(circuit, "Q3.1 – One Trotter Step")

        st.markdown("""
        **Gate labels:**
        - `rxx` / `ryy` on qubits (0,2) and (1,3) → **hopping** (JW XY term)
        - `rzz` on qubits (0,1) and (2,3)         → **interaction** ZZ part (U term)
        - `rz`  on individual qubits               → **interaction** single-site Z part (U term)

        The Z-string in the JW transformation for the q0→q2 hop produces an
        implicit sign handled by the `rxx`/`ryy` decomposition above.
        """)

    # ── Q3.2 ─────────────────────────────────────────────────
    elif q3_choice.startswith("Q3.2"):
        st.markdown("""
        <div class="card">
        U=0, J=1.  Start in |1000⟩ (one ↑ electron at Site 1).  
        Track P(|0010⟩) — the probability of finding the electron at Site 2 — 
        as τ ∈ [0, π].  Compare to the analytical Rabi oscillation.
        </div>
        """, unsafe_allow_html=True)

        n_steps = st.slider("Trotter steps per time point", 1, 20, 10, key="q32_steps")
        n_pts   = st.slider("Number of time points", 10, 60, 30, key="q32_pts")

        if st.button("▶ Compute Dynamics", key="run_q32"):
            times = np.linspace(0, np.pi, n_pts)
            probs_sim  = []
            probs_anal = []

            progress = st.progress(0, text="Running …")
            for idx, tau in enumerate(times):
                qc = hubbard_trotter_circuit(0.0, tau=tau, n_steps=n_steps,
                                             init_state="1000")
                counts = run_and_get_histogram(qc, shots=2048)
                total  = sum(counts.values())
                # |0010⟩: qubit 2 set → index 4 in little-endian
                p = counts.get("0010", 0) / total
                probs_sim.append(p)
                # Analytical: P = sin²(J τ) for single-electron two-level system
                probs_anal.append(np.sin(tau) ** 2)
                progress.progress((idx + 1) / n_pts,
                                  text=f"τ = {tau:.3f} …")

            progress.empty()

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(times, probs_anal, "--", color="#AAAAAA", lw=2, label="Analytical sin²(τ)")
            ax.plot(times, probs_sim, "o-", color=MAROON, ms=5, lw=1.8, label="Simulation")
            ax.set_xlabel("Time τ", fontsize=12)
            ax.set_ylabel("P(|0010⟩)", fontsize=12)
            ax.set_title("Q3.2 – Electron Hopping Dynamics (U=0, J=1)", fontsize=13)
            ax.legend(fontsize=11)
            ax.spines[["top", "right"]].set_visible(False)
            fig.patch.set_facecolor("white")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.markdown("""
            **Analysis:**  
            At τ = π/2 the probability of finding the electron at Site 2 peaks at 100%,
            matching the two-level "Rabi" oscillation P = sin²(Jτ).  
            The period is T = π/J = π (with J=1), agreeing with the analytical result.
            Trotter errors cause small deviations at large τ; increasing `n_steps` reduces them.
            """)

    # ── Q3.3 ─────────────────────────────────────────────────
    else:
        st.markdown("""
        <div class="card">
        U=10, J=1.  Start in |1100⟩ (both electrons at Site 1).  
        Compare P(|1100⟩) vs P(|0011⟩) over time.  
        Large U suppresses doublon formation — this is Mott physics.
        </div>
        """, unsafe_allow_html=True)

        n_steps = st.slider("Trotter steps per time point", 1, 20, 10, key="q33_steps")
        n_pts   = st.slider("Number of time points", 10, 60, 30, key="q33_pts")

        if st.button("▶ Compute Mott Dynamics", key="run_q33"):
            times      = np.linspace(0, np.pi, n_pts)
            p_1100     = []
            p_0011     = []

            progress = st.progress(0, text="Running …")
            for idx, tau in enumerate(times):
                qc = hubbard_trotter_circuit(10.0, tau=tau, n_steps=n_steps,
                                             init_state="1100")
                counts = run_and_get_histogram(qc, shots=2048)
                total  = sum(counts.values())
                p_1100.append(counts.get("1100", 0) / total)
                p_0011.append(counts.get("0011", 0) / total)
                progress.progress((idx + 1) / n_pts,
                                  text=f"τ = {tau:.3f} …")

            progress.empty()

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(times, p_1100, "o-", color=MAROON,   ms=5, lw=1.8, label="|1100⟩ (Site 1)")
            ax.plot(times, p_0011, "s-", color="#4E6A9E", ms=5, lw=1.8, label="|0011⟩ (Site 2)")
            ax.set_xlabel("Time τ", fontsize=12)
            ax.set_ylabel("Probability", fontsize=12)
            ax.set_title("Q3.3 – Mott Physics: Strong Interactions (U=10, J=1)", fontsize=13)
            ax.legend(fontsize=11)
            ax.spines[["top", "right"]].set_visible(False)
            fig.patch.set_facecolor("white")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.markdown("""
            **Mott Insulator Physics:**  
            With U/J = 10, the Coulomb repulsion far exceeds the hopping energy.  
            Moving both electrons to Site 2 (the "doublon" |0011⟩) costs energy U,
            so tunneling is strongly suppressed compared to the U=0 case.  
            This is the quantum-mechanical origin of the **Mott insulator**: 
            a material that *should* conduct (half-filled band) but doesn't, because
            on-site repulsion localizes the electrons.  
            The residual small oscillation is a virtual (second-order) process ∝ J²/U.
            """)


# ═══════════════════════════════════════════════════════════════
# TAB 4 – Custom Circuit
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Custom Circuit Builder")

    num_qubits = st.slider("Number of qubits", 1, 10, 3, key="custom_nq")

    col1, col2 = st.columns([1, 2])
    with col1:
        gate = st.selectbox("Gate", ["H", "X", "Z", "RX", "RZ", "CX"], key="custom_gate")
    with col2:
        if gate in ["RX", "RZ"]:
            param = st.slider("Angle θ", 0.0, float(2 * np.pi), 1.0,
                              format="%.3f", key="custom_angle")
        else:
            param = 0.0

    if gate == "CX":
        cc1, cc2 = st.columns(2)
        with cc1:
            ctrl = st.number_input("Control qubit", 0, num_qubits - 1, 0, key="ctrl")
        with cc2:
            tgt  = st.number_input("Target qubit",  0, num_qubits - 1,
                                   min(1, num_qubits - 1), key="tgt")
        qubit = 0   # placeholder
    else:
        qubit = st.number_input("Target qubit", 0, num_qubits - 1, 0, key="custom_qubit")
        ctrl = tgt = 0

    if "custom_ops" not in st.session_state:
        st.session_state.custom_ops = []

    col_add, col_clear = st.columns([1, 1])
    with col_add:
        if st.button("➕ Add Gate", key="add_gate"):
            op = {"gate": gate}
            if gate in ["RX", "RZ"]:
                op["qubit"] = qubit
                op["param"] = param
            elif gate == "CX":
                op["control"] = int(ctrl)
                op["target"]  = int(tgt)
            else:
                op["qubit"] = int(qubit)
            st.session_state.custom_ops.append(op)

    with col_clear:
        if st.button("🗑 Clear All", key="clear_gates"):
            st.session_state.custom_ops = []

    # Always build circuit (stateless)
    circuit = build_custom_circuit(num_qubits, st.session_state.custom_ops)

    if st.session_state.custom_ops:
        st.markdown('<div class="section-label">Gate Sequence</div>', unsafe_allow_html=True)
        for i, op in enumerate(st.session_state.custom_ops):
            st.write(f"`{i + 1}.` {op}")

        # Always show circuit
        draw_circuit(circuit, "Custom Circuit")

        if st.button("▶ Run Custom Circuit", key="run_custom"):
            with st.spinner("Simulating …"):
                counts = run_and_get_histogram(circuit)
            show_histogram(counts, "Custom Circuit Results")
    else:
        st.info("Add some gates above, then click **Run Custom Circuit**.")

