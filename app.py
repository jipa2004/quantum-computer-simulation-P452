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

from quantum_simulator_backend import run_and_get_histogram, run_single_shot
from circuits import (
    alice_circuit,
    bob_circuit,
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
        3-qubit teleportation protocol via a classical channel.<br>
        <b>q0</b> = Alice's qubit (state to teleport) &nbsp;·&nbsp;
        <b>q1</b> = Alice's Bell-pair qubit &nbsp;·&nbsp;
        <b>q2</b> = Bob's qubit.<br>
        The preset circuit shows Alice's side only. After she measures q0 and q1,
        the results are sent to Bob over a classical channel: Bob applies
        <b>Z</b> if q0 = 1, and <b>X</b> if q1 = 1.
        </div>
        """, unsafe_allow_html=True)

        col_l, col_r = st.columns([1, 2])
        with col_l:
            theta = st.slider(
                "Rotation angle θ (Alice's state)",
                0.0, float(2 * np.pi), 0.0,
                step=0.01, format="%.3f",
                key="tele_theta",
            )
            st.caption(f"θ = {theta:.4f} rad  ({theta / np.pi:.3f} π)")
            st.caption(f"|q0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩")

        # Always show Alice's preset circuit
        qc_alice = alice_circuit(theta)
        with col_r:
            draw_circuit(qc_alice, "Alice's Circuit (preset)")

        st.markdown("---")

        # ── N-shot teleportation simulation ──────
        st.markdown('<div class="section-label">Run teleportation</div>',
                    unsafe_allow_html=True)

        n_runs = st.slider("Number of teleportation runs (N)", 1, 1024, 100,
                           key="tele_n_runs")

        if st.button("▶ Run N Teleportations", key="run_tele"):
            # ── Phase 1: run Alice's circuit N times, collect per-shot results ──
            alice_counts: dict = {}   # {"m0 m1": count}
            last_m0, last_m1 = 0, 0

            with st.spinner(f"Running Alice's circuit {n_runs} times …"):
                raw = run_and_get_histogram(qc_alice, shots=n_runs)
                # Qiskit bitstring is MSB first: "m1 m0" → we want m0 (bit 0) and m1 (bit 1)
                for bitstring, count in raw.items():
                    # bitstring is "XY" where X=c1 (q1 result), Y=c0 (q0 result)
                    # (Qiskit orders classical bits MSB-first in the string)
                    m1 = int(bitstring[0])   # classical bit 1 → q1 measurement
                    m0 = int(bitstring[1])   # classical bit 0 → q0 measurement
                    key = f"q0={m0}, q1={m1}"
                    alice_counts[key] = alice_counts.get(key, 0) + count
                    last_m0, last_m1 = m0, m1   # just keep whichever is last iterated

                # Find the most frequent outcome as "last" to display
                most_common = max(raw.items(), key=lambda x: x[1])
                mc_str = most_common[0]
                last_m1 = int(mc_str[0])
                last_m0 = int(mc_str[1])

            # ── Display Alice's measurement counts ────────────────
            st.markdown('<div class="section-label">Alice\'s measurement results</div>',
                        unsafe_allow_html=True)

            acol1, acol2, acol3, acol4 = st.columns(4)
            for (label, cnt), col in zip(alice_counts.items(),
                                         [acol1, acol2, acol3, acol4]):
                col.metric(label, f"{cnt} shots", f"{cnt/n_runs*100:.1f}%")

            with st.expander("Raw counts from Alice's circuit"):
                st.write(raw)

            st.markdown("---")

            # ── Classical channel: show corrections for most common outcome ──
            st.markdown('<div class="section-label">Classical channel → Bob\'s corrections</div>',
                        unsafe_allow_html=True)

            st.markdown(f"""
            Most frequent outcome: **q0 = {last_m0}, q1 = {last_m1}**
            """)

            cc1, cc2 = st.columns(2)
            cc1.markdown(
                f"**Z gate on q2?** &nbsp; {'✅ Yes (q0 = 1)' if last_m0 == 1 else '❌ No (q0 = 0)'}",
                unsafe_allow_html=True,
            )
            cc2.markdown(
                f"**X gate on q2?** &nbsp; {'✅ Yes (q1 = 1)' if last_m1 == 1 else '❌ No (q1 = 0)'}",
                unsafe_allow_html=True,
            )

            st.markdown("---")

            # ── Phase 2: Bob's final state probabilities ──────────
            # Run Bob's corrected circuit for each of the four (m0,m1) outcomes,
            # weighted by how often Alice observed each.
            st.markdown('<div class="section-label">Bob\'s final measurement</div>',
                        unsafe_allow_html=True)

            bob_0_total = 0
            bob_1_total = 0

            for bitstring, count in raw.items():
                m1 = int(bitstring[0])
                m0 = int(bitstring[1])
                qc_bob = bob_circuit(theta, m0, m1)
                bob_raw = run_and_get_histogram(qc_bob, shots=count if count > 0 else 1)
                bob_0_total += bob_raw.get("0", 0)
                bob_1_total += bob_raw.get("1", 0)

            total_bob = bob_0_total + bob_1_total
            p0 = bob_0_total / total_bob if total_bob > 0 else 0
            p1 = bob_1_total / total_bob if total_bob > 0 else 0

            bc1, bc2 = st.columns(2)
            bc1.metric("P(Bob measures |0⟩)", f"{p0 * 100:.1f}%")
            bc2.metric("P(Bob measures |1⟩)", f"{p1 * 100:.1f}%")

            expected_0 = np.cos(theta / 2) ** 2
            expected_1 = np.sin(theta / 2) ** 2

            # Bob's histogram
            fig, ax = plt.subplots(figsize=(4, 3))
            bars = ax.bar(["Bob |0⟩", "Bob |1⟩"], [p0, p1],
                          color=MAROON, edgecolor="white", width=0.5)
            ax.axhline(expected_0, color="#AAAAAA", ls="--", lw=1.2,
                       label=f"Expected |0⟩ = {expected_0:.3f}")
            ax.axhline(expected_1, color="#888888", ls=":",  lw=1.2,
                       label=f"Expected |1⟩ = {expected_1:.3f}")
            for bar, p in zip(bars, [p0, p1]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{p:.3f}", ha="center", va="bottom", fontsize=10, color=DARK)
            ax.set_ylim(0, 1.15)
            ax.set_ylabel("Probability", fontsize=11)
            ax.legend(fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_facecolor("#FAFAFA")
            fig.patch.set_facecolor("white")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

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
    st.markdown("### Custom Circuit Builder")

    num_qubits = st.slider("Number of qubits", 1, 10, 3, key="custom_nq")

    st.markdown("---")

    # ── Gate catalogue ────────────────────────────
    SINGLE_QUBIT_GATES = ["H", "X", "Z", "RX", "RY", "RZ"]
    TWO_QUBIT_GATES    = ["CX", "SWAP"]
    ALL_GATES          = SINGLE_QUBIT_GATES + TWO_QUBIT_GATES

    col_gate, col_param = st.columns([1, 2])
    with col_gate:
        gate = st.selectbox("Gate", ALL_GATES, key="custom_gate")

    with col_param:
        if gate in ["RX", "RY", "RZ"]:
            param = st.slider(
                "Angle θ", 0.0, float(2 * np.pi), 1.0,
                format="%.3f", key="custom_angle",
            )
        else:
            param = 0.0
            st.write("")   # keep row height consistent

    # ── Qubit selector ────────────────────────────
    if gate in TWO_QUBIT_GATES:
        cc1, cc2 = st.columns(2)
        label_a  = "Control qubit" if gate == "CX" else "Qubit A"
        label_b  = "Target qubit"  if gate == "CX" else "Qubit B"
        with cc1:
            ctrl = int(st.number_input(label_a, 0, num_qubits - 1, 0, key="custom_ctrl"))
        with cc2:
            tgt  = int(st.number_input(label_b, 0, num_qubits - 1,
                                        min(1, num_qubits - 1), key="custom_tgt"))
        qubit = 0
    else:
        qubit = int(st.number_input("Target qubit", 0, num_qubits - 1, 0, key="custom_qubit"))
        ctrl = tgt = 0

    # ── Add / Clear buttons ───────────────────────
    if "custom_ops" not in st.session_state:
        st.session_state.custom_ops = []

    col_add, col_clear, _ = st.columns([1, 1, 4])
    with col_add:
        if st.button("➕ Add Gate", key="add_gate"):
            op = {"gate": gate}
            if gate in ["RX", "RY", "RZ"]:
                op["qubit"] = qubit
                op["param"] = param
            elif gate in TWO_QUBIT_GATES:
                op["control"] = ctrl
                op["target"]  = tgt
            else:
                op["qubit"] = qubit
            st.session_state.custom_ops.append(op)

    with col_clear:
        if st.button("🗑 Clear All", key="clear_gates"):
            st.session_state.custom_ops = []

    # ── Gate sequence + circuit preview ──────────
    if st.session_state.custom_ops:
        st.markdown('<div class="section-label">Gate Sequence</div>',
                    unsafe_allow_html=True)

        rows = []
        for i, op in enumerate(st.session_state.custom_ops):
            g = op["gate"]
            if g in ["RX", "RY", "RZ"]:
                target = f"q{op['qubit']}"
                detail = f"θ = {op['param']:.3f} rad"
            elif g in ["CX", "SWAP"]:
                target = f"q{op['control']} → q{op['target']}"
                detail = "3 CNOTs" if g == "SWAP" else ""
            else:
                target = f"q{op['qubit']}"
                detail = ""
            rows.append({"#": i + 1, "Gate": g, "Qubit(s)": target, "Detail": detail})

        st.table(rows)

        circuit = build_custom_circuit(num_qubits, st.session_state.custom_ops)
        draw_circuit(circuit, "Circuit Preview")

        if st.button("▶ Run Simulation", key="run_custom"):
            with st.spinner("Simulating …"):
                counts = run_and_get_histogram(circuit)
            show_histogram(counts, "Custom Circuit Results")

    else:
        st.info("Select a gate and qubit above, then click **➕ Add Gate** to start building.")
