"""
P452 Quantum Circuit Explorer
==============================
Streamlit frontend for the 10-qubit Qiskit-Aer simulator.
Theme: maroon + white  (set via .streamlit/config.toml)
Two tabs: Preset Circuits | Custom Circuit
"""

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantum_simulator_backend import run_and_get_histogram
from circuits import (
    teleportation_circuit,
    hubbard_trotter_circuit,
    build_custom_circuit,
)

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="P452 Quantum Explorer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MAROON = "#800000"
CREAM  = "#FFF8F0"
DARK   = "#1A0000"

# ─────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] {{
      font-family: 'EB Garamond', serif;
      background-color: #FFFFFF;
      color: {DARK};
  }}

  /* ── Header ── */
  .app-header {{
      background: {MAROON};
      color: white;
      padding: 1.4rem 2rem 1rem;
      border-radius: 0 0 12px 12px;
      margin-bottom: 1.4rem;
  }}
  .app-header h1 {{ font-size: 2rem; font-weight: 700; letter-spacing: 0.04em; margin: 0; }}
  .app-header p  {{ font-size: 1rem; opacity: 0.85; margin: 0.25rem 0 0; }}

  /* ── Section pill labels ── */
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

  /* ── Info card ── */
  .card {{
      background: {CREAM};
      border: 1px solid #E8D8D0;
      border-left: 4px solid {MAROON};
      border-radius: 8px;
      padding: 1rem 1.4rem;
      margin-bottom: 1rem;
  }}

  /* ── Buttons ── */
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
  .stButton > button:hover {{ opacity: 0.82 !important; }}

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab"] {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.85rem;
      letter-spacing: 0.05em;
  }}
  .stTabs [aria-selected="true"] {{
      border-bottom: 3px solid {MAROON} !important;
      color: {MAROON} !important;
  }}

  hr {{ border-color: #E0C0C0; }}
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
# Shared helpers
# ─────────────────────────────────────────────────────────────

def draw_circuit(qc, title="Circuit Diagram"):
    st.markdown(f'<div class="section-label">🔬 {title}</div>', unsafe_allow_html=True)
    style = {
        "backgroundcolor":  "#FFFFFF",
        "linecolor":        MAROON,
        "textcolor":        DARK,
        "gatefacecolor":    "#FDECEA",
        "gateoutlinecolor": MAROON,
    }
    fig = qc.draw(output="mpl", style=style, fold=-1)
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def show_histogram(counts: dict, title="Measurement Results"):
    st.markdown(f'<div class="section-label">📊 {title}</div>', unsafe_allow_html=True)

    labels = sorted(counts.keys())
    values = [counts[k] for k in labels]
    total  = sum(values)
    probs  = [v / total for v in values]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.65), 3.6))
    bars = ax.bar(labels, probs, color=MAROON, edgecolor="white", width=0.6)
    ax.set_ylabel("Probability", fontsize=11)
    ax.set_xlabel("Bitstring", fontsize=11)
    ax.set_ylim(0, min(1.18, max(probs) * 1.28))
    ax.tick_params(axis="x", rotation=45)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")
    for bar, p in zip(bars, probs):
        if p > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{p:.2f}", ha="center", va="bottom", fontsize=8, color=DARK)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    with st.expander("Raw counts"):
        st.write(counts)


# ═══════════════════════════════════════════════════════════════
# Two tabs
# ═══════════════════════════════════════════════════════════════
tab_preset, tab_custom = st.tabs(["📡  Preset Circuits", "🎛  Custom Circuit"])


# ═══════════════════════════════════════════════════════════════
# TAB 1 – PRESET CIRCUITS
# ═══════════════════════════════════════════════════════════════
with tab_preset:

    preset = st.radio(
        "Choose a preset circuit",
        ["Teleportation", "Hubbard Model"],
        horizontal=True,
        key="preset_choice",
    )

    st.markdown("---")

    # ─────────────────────────────────────────────
    # TELEPORTATION
    # ─────────────────────────────────────────────
    if preset == "Teleportation":
        st.markdown("### Quantum Teleportation")
        st.markdown("""
        <div class="card">
        3-qubit teleportation protocol.<br>
        <b>q0</b> = Alice's qubit (state to teleport) &nbsp;·&nbsp;
        <b>q1</b> = Alice's Bell-pair qubit &nbsp;·&nbsp;
        <b>q2</b> = Bob's qubit.<br>
        Adjust θ to change the state Alice sends. At θ = 0 Alice sends |0⟩;
        at θ = π she sends |1⟩.
        </div>
        """, unsafe_allow_html=True)

        col_l, col_r = st.columns([1, 2])
        with col_l:
            theta = st.slider(
                "Rotation angle θ (Alice's qubit)",
                0.0, float(2 * np.pi), 0.0,
                step=0.01, format="%.3f",
                key="tele_theta",
            )
            st.caption(f"θ = {theta:.4f} rad  ({theta / np.pi:.3f} π)")
            st.caption(f"|q0⟩ ∝ cos(θ/2)|0⟩ + sin(θ/2)|1⟩")

        circuit = teleportation_circuit(theta)

        with col_r:
            draw_circuit(circuit, "Teleportation Circuit")

        if st.button("▶ Run Simulation (1024 shots)", key="run_tele"):
            with st.spinner("Simulating …"):
                counts = run_and_get_histogram(circuit, shots=1024)
            show_histogram(counts, "Teleportation – Measurement Results")

            # Bob's qubit is classical bit 2; Qiskit prints MSB first,
            # so it appears as the last character of the bitstring.
            bob_0 = sum(v for k, v in counts.items() if k[-1] == "0")
            bob_1 = sum(v for k, v in counts.items() if k[-1] == "1")
            total  = bob_0 + bob_1
            c1, c2 = st.columns(2)
            c1.metric("P(Bob = |0⟩)", f"{bob_0 / total * 100:.1f}%")
            c2.metric("P(Bob = |1⟩)", f"{bob_1 / total * 100:.1f}%")

            expected_0 = np.cos(theta / 2) ** 2
            st.markdown(f"""
            **Expected (ideal):** P(|0⟩) = cos²(θ/2) = {expected_0:.3f},  
            P(|1⟩) = sin²(θ/2) = {1 - expected_0:.3f}.  
            Any deviation from ideal is shot noise (≈ 1/√1024 ≈ 3%).
            """)

    # ─────────────────────────────────────────────
    # HUBBARD MODEL
    # ─────────────────────────────────────────────
    else:
        st.markdown("### Fermi-Hubbard Model (Trotterized)")
        st.markdown("""
        <div class="card">
        4-qubit 2-site Hubbard model via Jordan-Wigner mapping + Trotter decomposition.<br>
        <b>q0</b> = site 1 ↑ &nbsp;·&nbsp; <b>q1</b> = site 1 ↓ &nbsp;·&nbsp;
        <b>q2</b> = site 2 ↑ &nbsp;·&nbsp; <b>q3</b> = site 2 ↓
        </div>
        """, unsafe_allow_html=True)

        sub = st.selectbox(
            "Checkpoint",
            ["Q3.1 – Circuit Architecture (1 Trotter step)",
             "Q3.2 – Non-Interacting Dynamics (U = 0)",
             "Q3.3 – Mott Physics (U = 10)"],
            key="hubbard_sub",
        )

        st.markdown("---")

        # ── Q3.1 ──────────────────────────────────
        if sub.startswith("Q3.1"):
            st.markdown("""
            One Trotter step of the Hubbard Hamiltonian.  
            **Hopping term**: `Rxx` + `Ryy` on (q0,q2) and (q1,q3).  
            **Interaction term**: `Rzz` + single-qubit `Rz` on (q0,q1) and (q2,q3).
            """)
            col1, col2 = st.columns(2)
            with col1:
                uj  = st.slider("U/J", 0.0, 10.0, 1.0, step=0.1, key="q31_uj")
            with col2:
                tau = st.slider("τ (time step)", 0.01, float(np.pi), 1.0,
                                step=0.01, key="q31_tau")

            circuit = hubbard_trotter_circuit(uj, tau=tau, n_steps=1, init_state="1000")
            draw_circuit(circuit, "Q3.1 – One Trotter Step")

            st.markdown("""
            **Gate key:**  
            `rxx` / `ryy` → hopping (JW XY term) &nbsp;·&nbsp;
            `rzz` → interaction ZZ part &nbsp;·&nbsp;
            `rz` → single-site Z energy shifts
            """)

        # ── Q3.2 ──────────────────────────────────
        elif sub.startswith("Q3.2"):
            st.markdown("""
            **U = 0, J = 1.** Initial state: |1000⟩ (one ↑ electron at Site 1).  
            Plot P(|0010⟩) — probability of finding the electron at Site 2 — vs τ ∈ [0, π].
            """)
            col1, col2 = st.columns(2)
            with col1:
                n_steps = st.slider("Trotter steps per time point", 1, 20, 10, key="q32_steps")
            with col2:
                n_pts = st.slider("Number of time points", 10, 60, 30, key="q32_pts")

            if st.button("▶ Compute Dynamics", key="run_q32"):
                times      = np.linspace(0, np.pi, n_pts)
                probs_sim  = []
                probs_anal = []
                progress   = st.progress(0, text="Running …")

                for i, tau in enumerate(times):
                    qc = hubbard_trotter_circuit(0.0, tau=tau, n_steps=n_steps,
                                                 init_state="1000")
                    counts = run_and_get_histogram(qc, shots=2048)
                    total  = sum(counts.values())
                    probs_sim.append(counts.get("0010", 0) / total)
                    probs_anal.append(np.sin(tau) ** 2)
                    progress.progress((i + 1) / n_pts, text=f"τ = {tau:.3f} …")

                progress.empty()

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(times, probs_anal, "--", color="#AAAAAA", lw=2,
                        label="Analytical sin²(τ)")
                ax.plot(times, probs_sim, "o-", color=MAROON, ms=5, lw=1.8,
                        label="Simulation")
                ax.set_xlabel("Time τ", fontsize=12)
                ax.set_ylabel("P(|0010⟩)", fontsize=12)
                ax.set_title("Q3.2 – Electron Hopping (U=0, J=1)", fontsize=13)
                ax.legend(fontsize=11)
                ax.spines[["top", "right"]].set_visible(False)
                fig.patch.set_facecolor("white")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                st.markdown("""
                **Analysis:** Peak at τ = π/2 confirms complete transfer to Site 2,
                matching the Rabi oscillation P = sin²(Jτ), period T = π/J = π.  
                Trotter error shrinks as `n_steps` increases.
                """)

        # ── Q3.3 ──────────────────────────────────
        else:
            st.markdown("""
            **U = 10, J = 1.** Initial state: |1100⟩ (both electrons at Site 1).  
            Compare P(|1100⟩) vs P(|0011⟩) over time — Mott insulator physics.
            """)
            col1, col2 = st.columns(2)
            with col1:
                n_steps = st.slider("Trotter steps per time point", 1, 20, 10, key="q33_steps")
            with col2:
                n_pts = st.slider("Number of time points", 10, 60, 30, key="q33_pts")

            if st.button("▶ Compute Mott Dynamics", key="run_q33"):
                times    = np.linspace(0, np.pi, n_pts)
                p_1100   = []
                p_0011   = []
                progress = st.progress(0, text="Running …")

                for i, tau in enumerate(times):
                    qc = hubbard_trotter_circuit(10.0, tau=tau, n_steps=n_steps,
                                                 init_state="1100")
                    counts = run_and_get_histogram(qc, shots=2048)
                    total  = sum(counts.values())
                    p_1100.append(counts.get("1100", 0) / total)
                    p_0011.append(counts.get("0011", 0) / total)
                    progress.progress((i + 1) / n_pts, text=f"τ = {tau:.3f} …")

                progress.empty()

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(times, p_1100, "o-", color=MAROON,    ms=5, lw=1.8,
                        label="|1100⟩ (Site 1)")
                ax.plot(times, p_0011, "s-", color="#4E6A9E", ms=5, lw=1.8,
                        label="|0011⟩ (Site 2 doublon)")
                ax.set_xlabel("Time τ", fontsize=12)
                ax.set_ylabel("Probability", fontsize=12)
                ax.set_title("Q3.3 – Mott Physics (U=10, J=1)", fontsize=13)
                ax.legend(fontsize=11)
                ax.spines[["top", "right"]].set_visible(False)
                fig.patch.set_facecolor("white")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                st.markdown("""
                **Mott Insulator Physics:** With U/J = 10, on-site Coulomb repulsion far
                exceeds hopping energy. Moving both electrons to Site 2 costs U, so
                tunneling is strongly suppressed. The residual oscillation is a virtual
                second-order process ∝ J²/U — the hallmark of a Mott insulator.
                """)


# ═══════════════════════════════════════════════════════════════
# TAB 2 – CUSTOM CIRCUIT
# ═══════════════════════════════════════════════════════════════
with tab_custom:

    # ── Constants ─────────────────────────────────
    SINGLE_QUBIT_GATES = ["H", "X", "Z", "RX", "RY", "RZ"]
    TWO_QUBIT_GATES    = ["CX", "SWAP"]
    ALL_GATES          = SINGLE_QUBIT_GATES + TWO_QUBIT_GATES

    # ── Session state ─────────────────────────────
    if "custom_ops" not in st.session_state:
        st.session_state.custom_ops = []

    # ── Helper: one-line description of a gate op ─
    def op_label(op):
        g = op["gate"]
        if g in ["RX", "RY", "RZ"]:
            return f"{g}(θ={op['param']:.3f})  q{op['qubit']}"
        elif g in ["CX", "SWAP"]:
            return f"{g}  q{op['control']} → q{op['target']}"
        else:
            return f"{g}  q{op['qubit']}"

    # ── Helper: build an op dict from current widget values ──
    def make_op(gate, qubit, ctrl, tgt, param):
        op = {"gate": gate}
        if gate in ["RX", "RY", "RZ"]:
            op["qubit"] = qubit
            op["param"] = param
        elif gate in ["CX", "SWAP"]:
            op["control"] = ctrl
            op["target"]  = tgt
        else:
            op["qubit"] = qubit
        return op

    # ─────────────────────────────────────────────────────────────
    # Two-column layout: left = controls, right = live circuit
    # ─────────────────────────────────────────────────────────────
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown("### Custom Circuit Builder")

        num_qubits = st.slider("Number of qubits", 1, 10, 3, key="custom_nq")
        st.markdown("---")

        # ── Gate picker ───────────────────────────
        st.markdown('<div class="section-label">Add a gate</div>', unsafe_allow_html=True)

        gate = st.selectbox("Gate type", ALL_GATES, key="custom_gate")

        if gate in ["RX", "RY", "RZ"]:
            param = st.slider("Angle θ", 0.0, float(2 * np.pi), 1.0,
                              format="%.3f", key="custom_angle")
        else:
            param = 0.0

        if gate in TWO_QUBIT_GATES:
            label_a = "Control qubit" if gate == "CX" else "Qubit A"
            label_b = "Target qubit"  if gate == "CX" else "Qubit B"
            ca, cb = st.columns(2)
            with ca:
                ctrl = int(st.number_input(label_a, 0, num_qubits - 1, 0,
                                           key="custom_ctrl"))
            with cb:
                tgt = int(st.number_input(label_b, 0, num_qubits - 1,
                                          min(1, num_qubits - 1), key="custom_tgt"))
            qubit = 0
        else:
            qubit = int(st.number_input("Target qubit", 0, num_qubits - 1, 0,
                                        key="custom_qubit"))
            ctrl = tgt = 0

        # Append-to-end
        if st.button("➕ Append Gate", key="add_gate", use_container_width=True):
            st.session_state.custom_ops.append(make_op(gate, qubit, ctrl, tgt, param))
            st.rerun()

        st.markdown("---")

        # ── Interactive gate sequence editor ──────
        ops = st.session_state.custom_ops

        if ops:
            st.markdown('<div class="section-label">Gate sequence</div>',
                        unsafe_allow_html=True)
            st.caption("↑ ↓ reorder · 🗑 delete · ＋ insert selected gate after this row")

            # Column headers
            hc = st.columns([2, 1, 1, 1, 1])
            for col, lbl in zip(hc, ["Gate", "↑", "↓", "🗑", "＋"]):
                col.markdown(f"<small><b>{lbl}</b></small>", unsafe_allow_html=True)

            action = None  # only one action fires per rerun

            for i, op in enumerate(ops):
                rc = st.columns([2, 1, 1, 1, 1])

                rc[0].markdown(
                    f"`{i+1}.` <span style='font-family:monospace;font-size:0.8rem'>"
                    f"{op_label(op)}</span>",
                    unsafe_allow_html=True,
                )

                if rc[1].button("↑", key=f"up_{i}",
                                disabled=(i == 0),
                                use_container_width=True):
                    action = ("move_up", i)

                if rc[2].button("↓", key=f"dn_{i}",
                                disabled=(i == len(ops) - 1),
                                use_container_width=True):
                    action = ("move_down", i)

                if rc[3].button("🗑", key=f"del_{i}",
                                use_container_width=True):
                    action = ("delete", i)

                if rc[4].button("＋", key=f"ins_{i}",
                                use_container_width=True):
                    action = ("insert_after", i)

            # Apply the single action that fired this rerun
            if action:
                kind, idx = action
                if kind == "move_up":
                    ops[idx - 1], ops[idx] = ops[idx], ops[idx - 1]
                elif kind == "move_down":
                    ops[idx], ops[idx + 1] = ops[idx + 1], ops[idx]
                elif kind == "delete":
                    ops.pop(idx)
                elif kind == "insert_after":
                    ops.insert(idx + 1, make_op(gate, qubit, ctrl, tgt, param))
                st.rerun()

            st.markdown("---")
            if st.button("🗑 Clear All", key="clear_gates", use_container_width=True):
                st.session_state.custom_ops = []
                st.rerun()

        else:
            st.info("Click **Append Gate** to start building your circuit.")

    # ─────────────────────────────────────────────
    # Right panel: live circuit preview + run
    # ─────────────────────────────────────────────
    with right:
        ops = st.session_state.custom_ops
        if ops:
            circuit = build_custom_circuit(num_qubits, ops)
            draw_circuit(circuit, "Circuit Preview")

            st.markdown("")
            if st.button("▶ Run Simulation", key="run_custom", use_container_width=True):
                with st.spinner("Simulating …"):
                    counts = run_and_get_histogram(circuit)
                show_histogram(counts, "Measurement Results")
        else:
            st.markdown("""
            <div class="card" style="margin-top:3rem; text-align:center; color:#999;">
              <p style="font-size:1.1rem;">Your circuit will appear here.</p>
              <p>Add gates using the panel on the left.</p>
            </div>
            """, unsafe_allow_html=True)

