"""
P452 Quantum Circuit Explorer
==============================
Streamlit frontend for the 10-qubit Qiskit-Aer simulator.
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
    full_teleportation_circuit,
    hubbard_trotter_circuit,
    build_custom_circuit,
)

st.set_page_config(
    page_title="P452 Quantum Explorer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MAROON = "#800000"
CREAM  = "#FFF8F0"
DARK   = "#1A0000"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
  html, body, [class*="css"] {{
      font-family: 'EB Garamond', serif;
      background-color: #FFFFFF;
      color: {DARK};
  }}
  .app-header {{
      background: {MAROON}; color: white;
      padding: 1.4rem 2rem 1rem;
      border-radius: 0 0 12px 12px; margin-bottom: 1.4rem;
  }}
  .app-header h1 {{ font-size: 2rem; font-weight: 700; letter-spacing: 0.04em; margin: 0; }}
  .app-header p  {{ font-size: 1rem; opacity: 0.85; margin: 0.25rem 0 0; }}
  .section-label {{
      display: inline-block; background: {MAROON}; color: white;
      font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
      letter-spacing: 0.12em; padding: 0.2rem 0.7rem;
      border-radius: 4px; margin-bottom: 0.5rem; text-transform: uppercase;
  }}
  .card {{
      background: {CREAM}; border: 1px solid #E8D8D0;
      border-left: 4px solid {MAROON}; border-radius: 8px;
      padding: 1rem 1.4rem; margin-bottom: 1rem;
  }}
  .stButton > button {{
      background: {MAROON} !important; color: white !important;
      border: none !important; border-radius: 6px !important;
      font-family: 'EB Garamond', serif !important; font-size: 1rem !important;
      padding: 0.45rem 1.4rem !important; transition: opacity 0.2s;
  }}
  .stButton > button:hover {{ opacity: 0.82 !important; }}
  .stTabs [data-baseweb="tab"] {{
      font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; letter-spacing: 0.05em;
  }}
  .stTabs [aria-selected="true"] {{
      border-bottom: 3px solid {MAROON} !important; color: {MAROON} !important;
  }}
  hr {{ border-color: #E0C0C0; }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="app-header">
  <h1>&#9883;&#65039; P452 Quantum Circuit Explorer</h1>
  <p>10-Qubit Simulator &middot; Qiskit-Aer Backend &middot; University of Chicago</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────

def draw_circuit(qc, title="Circuit Diagram"):
    st.markdown(f'<div class="section-label">&#128300; {title}</div>', unsafe_allow_html=True)
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
    st.markdown(f'<div class="section-label">&#128202; {title}</div>', unsafe_allow_html=True)
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


tab_preset, tab_custom = st.tabs(["Preset Circuits", "Custom Circuit"])


# ═══════════════════════════════════════════════════════════════
# TAB 1 – PRESET CIRCUITS
# ═══════════════════════════════════════════════════════════════
with tab_preset:

    preset = st.radio(
        "Choose a preset circuit",
        ["Teleportation", "Hubbard Model"],
        horizontal=True, key="preset_choice",
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
        <b>q0</b> = Alice's qubit (state to teleport) &nbsp;&middot;&nbsp;
        <b>q1</b> = Alice's Bell-pair qubit &nbsp;&middot;&nbsp;
        <b>q2</b> = Bob's qubit.<br>
        The preset circuit shows Alice's side only. After she measures q0 and q1,
        the results are sent to Bob over a classical channel: Bob applies
        <b>Z</b> if q0 = 1, and <b>X</b> if q1 = 1.
        </div>
        """, unsafe_allow_html=True)

        col_l, col_r = st.columns([1, 2])
        with col_l:
            theta = st.slider(
                "Rotation angle theta (Alice's state)",
                0.0, float(2 * np.pi), 0.0,
                step=0.01, format="%.3f", key="tele_theta",
            )
            st.caption(f"theta = {theta:.4f} rad  ({theta / np.pi:.3f} pi)")
            st.caption(f"|q0> = cos(theta/2)|0> + sin(theta/2)|1>")

        qc_alice = alice_circuit(theta)
        with col_r:
            draw_circuit(qc_alice, "Alice's Circuit (preset)")

        st.markdown("---")
        st.markdown('<div class="section-label">Run teleportation</div>',
                    unsafe_allow_html=True)

        n_runs = st.slider("Number of teleportation runs (N)", 1, 1024, 100,
                           key="tele_n_runs")

        if st.button("Run N Teleportations", key="run_tele"):
            # Run the FULL circuit: Alice prep + mid-circuit measurement (collapse)
            # + classical feed-forward + Bob's corrections + Bob's measurement.
            # Aer correctly collapses q0/q1 before Bob's conditional gates fire,
            # giving the physically correct (not 50/50) result.
            qc_full = full_teleportation_circuit(theta)

            with st.spinner(f"Running {n_runs} teleportations ..."):
                raw = run_and_get_histogram(qc_full, shots=n_runs)

            # Bitstring from 3 separate 1-bit registers printed MSB-first with spaces:
            # "c2 c1 c0"  where c0=Alice q0, c1=Alice q1, c2=Bob q2
            alice_counts = {}
            bob_0_total  = 0
            bob_1_total  = 0

            for bitstring, count in raw.items():
                bits = bitstring.replace(" ", "")  # remove spaces between registers
                # After stripping: bits[0]=c2(Bob), bits[1]=c1(Alice q1), bits[2]=c0(Alice q0)
                bob_bit = int(bits[0])
                m1      = int(bits[1])
                m0      = int(bits[2])
                key = f"q0={m0}, q1={m1}"
                alice_counts[key] = alice_counts.get(key, 0) + count
                if bob_bit == 0:
                    bob_0_total += count
                else:
                    bob_1_total += count

            best    = max(alice_counts.items(), key=lambda x: x[1])[0]
            last_m0 = int(best.split("q0=")[1].split(",")[0])
            last_m1 = int(best.split("q1=")[1])

            # Alice's results
            st.markdown('<div class="section-label">Alice\'s measurement results</div>',
                        unsafe_allow_html=True)
            acols = st.columns(4)
            for (label, cnt), col in zip(alice_counts.items(), acols):
                col.metric(label, f"{cnt} shots", f"{cnt / n_runs * 100:.1f}%")
            with st.expander("Raw counts (full circuit)"):
                st.write(raw)

            st.markdown("---")

            # Classical channel
            st.markdown('<div class="section-label">Classical channel - Bob\'s corrections</div>',
                        unsafe_allow_html=True)
            st.markdown(f"Most frequent outcome: **q0 = {last_m0}, q1 = {last_m1}**")
            cc1, cc2 = st.columns(2)
            cc1.markdown(
                f"**Z gate on q2?** &nbsp; "
                f"{'Yes (q0 = 1)' if last_m0 == 1 else 'No (q0 = 0)'}",
                unsafe_allow_html=True,
            )
            cc2.markdown(
                f"**X gate on q2?** &nbsp; "
                f"{'Yes (q1 = 1)' if last_m1 == 1 else 'No (q1 = 0)'}",
                unsafe_allow_html=True,
            )

            st.markdown("---")

            # Bob's measurement
            st.markdown('<div class="section-label">Bob\'s final measurement</div>',
                        unsafe_allow_html=True)

            total_bob = bob_0_total + bob_1_total
            p0 = bob_0_total / total_bob if total_bob > 0 else 0
            p1 = bob_1_total / total_bob if total_bob > 0 else 0

            bc1, bc2 = st.columns(2)
            bc1.metric("P(Bob measures |0>)", f"{p0 * 100:.1f}%")
            bc2.metric("P(Bob measures |1>)", f"{p1 * 100:.1f}%")

            expected_0 = np.cos(theta / 2) ** 2
            expected_1 = np.sin(theta / 2) ** 2

            fig, ax = plt.subplots(figsize=(4, 3))
            bars = ax.bar(["Bob |0>", "Bob |1>"], [p0, p1],
                          color=MAROON, edgecolor="white", width=0.5)
            ax.axhline(expected_0, color="#AAAAAA", ls="--", lw=1.2,
                       label=f"Expected |0> = {expected_0:.3f}")
            ax.axhline(expected_1, color="#888888", ls=":",  lw=1.2,
                       label=f"Expected |1> = {expected_1:.3f}")
            for bar, p_val in zip(bars, [p0, p1]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{p_val:.3f}", ha="center", va="bottom", fontsize=10, color=DARK)
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
        4-qubit 2-site Fermi-Hubbard model via Jordan-Wigner mapping + first-order Trotter.<br>
        <b>q0</b> = site 1 &uarr; &nbsp;&middot;&nbsp; <b>q1</b> = site 1 &darr;
        &nbsp;&middot;&nbsp; <b>q2</b> = site 2 &uarr; &nbsp;&middot;&nbsp; <b>q3</b> = site 2 &darr;<br>
        H = &minus;(J/2)[X&#8320;Z&#8321;X&#8322; + Y&#8320;Z&#8321;Y&#8322; + X&#8321;Z&#8322;X&#8323; + Y&#8321;Z&#8322;Y&#8323;]
        &minus; (U/4)(Z&#8320;+Z&#8321;+Z&#8322;+Z&#8323;) + (U/4)(Z&#8320;Z&#8321; + Z&#8322;Z&#8323;)
        </div>
        """, unsafe_allow_html=True)

        # ── Parameters ────────────────────────────
        pc1, pc2, pc3, pc4 = st.columns(4)
        with pc1:
            J_val   = st.slider("J (hopping)", 0.0, 5.0, 1.0, step=0.1, key="hub_J")
        with pc2:
            U_val   = st.slider("U (interaction)", 0.0, 20.0, 4.0, step=0.5, key="hub_U")
        with pc3:
            tau_val = st.slider("τ (time step)", 0.01, float(np.pi), 1.0,
                                step=0.01, key="hub_tau")
        with pc4:
            n_steps_val = st.slider("Trotter steps", 1, 20, 1, key="hub_nsteps")

        st.markdown("---")

        # ── Initial state builder ─────────────────
        st.markdown('<div class="section-label">Initial State</div>', unsafe_allow_html=True)
        st.caption(
            "Build the initial 4-qubit state using the gate editor below. "
            "Leave empty for |0000⟩. "
            "Example: X on q0 gives |1000⟩ (one ↑ electron at Site 1)."
        )

        HUB_GATES     = ["X", "H", "RY", "RX", "RZ", "CX"]
        HUB_TWO_QUBIT = ["CX"]

        if "hub_init_ops" not in st.session_state:
            st.session_state.hub_init_ops = []

        def hub_op_label(op):
            g = op["gate"]
            if g in ["RX", "RY", "RZ"]:
                return f"{g}(θ={op['param']:.3f}) q{op['qubit']}"
            elif g == "CX":
                return f"CX q{op['control']}→q{op['target']}"
            else:
                return f"{g} q{op['qubit']}"

        def hub_make_op(gate, qubit, ctrl, tgt, param):
            op = {"gate": gate}
            if gate in ["RX", "RY", "RZ"]:
                op["qubit"] = qubit
                op["param"] = param
            elif gate == "CX":
                op["control"] = ctrl
                op["target"]  = tgt
            else:
                op["qubit"] = qubit
            return op

        init_left, init_right = st.columns([1, 2], gap="large")

        with init_left:
            hub_gate = st.selectbox("Gate", HUB_GATES, key="hub_gate")

            if hub_gate in ["RX", "RY", "RZ"]:
                hub_param = st.slider("Angle θ", 0.0, float(2 * np.pi), np.pi / 2,
                                      format="%.3f", key="hub_angle")
            else:
                hub_param = 0.0

            if hub_gate == "CX":
                hca, hcb = st.columns(2)
                with hca:
                    hub_ctrl = int(st.number_input("Control", 0, 3, 0, key="hub_ctrl"))
                with hcb:
                    hub_tgt  = int(st.number_input("Target",  0, 3, 1, key="hub_tgt"))
                hub_qubit = 0
            else:
                hub_qubit = int(st.number_input("Qubit (0–3)", 0, 3, 0, key="hub_qubit"))
                hub_ctrl = hub_tgt = 0

            hb1, hb2 = st.columns(2)
            with hb1:
                if st.button("Append", key="hub_add", use_container_width=True):
                    st.session_state.hub_init_ops.append(
                        hub_make_op(hub_gate, hub_qubit, hub_ctrl, hub_tgt, hub_param)
                    )
                    st.rerun()
            with hb2:
                if st.button("Clear", key="hub_clear", use_container_width=True):
                    st.session_state.hub_init_ops = []
                    st.rerun()

            hub_ops = st.session_state.hub_init_ops
            if hub_ops:
                st.markdown("**Gate sequence:**")
                hub_action = None
                hc = st.columns([2, 1, 1, 1, 1])
                for col, lbl in zip(hc, ["Gate", "↑", "↓", "🗑", "＋"]):
                    col.markdown(f"<small><b>{lbl}</b></small>", unsafe_allow_html=True)
                for i, op in enumerate(hub_ops):
                    rc = st.columns([2, 1, 1, 1, 1])
                    rc[0].markdown(
                        f"`{i+1}.` <span style='font-family:monospace;font-size:0.8rem'>"
                        f"{hub_op_label(op)}</span>",
                        unsafe_allow_html=True,
                    )
                    if rc[1].button("↑", key=f"hub_up_{i}", disabled=(i == 0),
                                    use_container_width=True):
                        hub_action = ("move_up", i)
                    if rc[2].button("↓", key=f"hub_dn_{i}",
                                    disabled=(i == len(hub_ops) - 1),
                                    use_container_width=True):
                        hub_action = ("move_down", i)
                    if rc[3].button("🗑", key=f"hub_del_{i}", use_container_width=True):
                        hub_action = ("delete", i)
                    if rc[4].button("＋", key=f"hub_ins_{i}", use_container_width=True):
                        hub_action = ("insert_after", i)

                if hub_action:
                    kind, idx = hub_action
                    if kind == "move_up":
                        hub_ops[idx-1], hub_ops[idx] = hub_ops[idx], hub_ops[idx-1]
                    elif kind == "move_down":
                        hub_ops[idx], hub_ops[idx+1] = hub_ops[idx+1], hub_ops[idx]
                    elif kind == "delete":
                        hub_ops.pop(idx)
                    elif kind == "insert_after":
                        hub_ops.insert(
                            idx + 1,
                            hub_make_op(hub_gate, hub_qubit, hub_ctrl, hub_tgt, hub_param)
                        )
                    st.rerun()

        with init_right:
            # Always show current Hubbard circuit (updates live with sliders + init state)
            qc_hub = hubbard_trotter_circuit(
                J=J_val, U=U_val, tau=tau_val,
                n_steps=n_steps_val,
                init_ops=st.session_state.hub_init_ops,
            )
            draw_circuit(qc_hub, "Hubbard Circuit (1 Trotter step shown)")
            st.markdown("""
            **Gate key:**  
            H / Rx(±π/2) sandwiches → basis change for X·Z·X / Y·Z·Y hopping strings  
            CX ladder + Rz → hopping Pauli string exp (θ_hop = J·τ/2·n_steps)  
            CX + Rz → ZZ interaction (θ_int = U·τ/2·n_steps)  
            Rz on each qubit → single-site Z shifts (−U/4 terms)
            """)

        st.markdown("---")

        # ── Single-time snapshot ──────────────────
        st.markdown('<div class="section-label">Snapshot at τ</div>',
                    unsafe_allow_html=True)
        n_shots = st.slider("Shots", 256, 4096, 1024, step=256, key="hub_shots")
        if st.button("Run Hubbard Simulation", key="run_hub"):
            with st.spinner("Simulating ..."):
                counts = run_and_get_histogram(qc_hub, shots=n_shots)
            show_histogram(counts, "Hubbard – Measurement Results")

        st.markdown("---")

        # ── Dynamics: P(state) vs time ────────────
        st.markdown('<div class="section-label">Time Evolution</div>',
                    unsafe_allow_html=True)
        st.caption(
            "Sweeps τ from 0 to the current slider value and plots the exact "
            "statevector probability of the selected states at each time point. "
            "Use ≥ 20 Trotter steps (dynamics) for accurate results — "
            "too few steps produce completely wrong physics regardless of U."
        )

        # All 16 possible 4-qubit bitstrings displayed q3q2q1q0 (MSB first)
        all_states = [format(i, "04b") for i in range(16)]

        dyn_col1, dyn_col2 = st.columns([1, 2])
        with dyn_col1:
            n_pts = st.slider("Time points", 10, 100, 40, key="hub_npts")
            dyn_steps = st.slider(
                "Trotter steps (dynamics)",
                min_value=5, max_value=200, value=50, step=5,
                key="hub_dyn_steps",
                help="Keep ≥ 20 for reliable results. The circuit diagram uses "
                     "the separate Trotter steps slider above.",
            )
            selected_states = st.multiselect(
                "States to plot",
                options=all_states,
                default=["0011", "1100"],
                key="hub_states",
            )

        if dyn_steps < 10:
            st.warning(
                f"⚠️ {dyn_steps} Trotter steps is too coarse — dynamics will be "
                "inaccurate. Use ≥ 20 for qualitatively correct results."
            )

        if st.button("Compute Dynamics", key="hub_dynamics"):
            if not selected_states:
                st.warning("Select at least one state to plot.")
            else:
                times = np.linspace(0.0, tau_val, n_pts)
                probs_t = {s: [] for s in selected_states}

                progress = st.progress(0, text="Computing statevectors ...")
                for ti, t in enumerate(times):
                    # Use tiny nonzero tau at t=0 so circuit is valid
                    tau_t = t if t > 1e-10 else 1e-10
                    qc_t = hubbard_trotter_circuit(
                        J=J_val, U=U_val, tau=tau_t,
                        n_steps=dyn_steps,
                        init_ops=st.session_state.hub_init_ops,
                    )

                    from quantum_simulator_backend import get_statevector
                    sv = get_statevector(qc_t)
                    probs_all = np.abs(sv) ** 2

                    # Displayed string s is q3q2q1q0 (MSB first).
                    # Qiskit SV: q0=LSB, so int(s,2) maps correctly:
                    # "1100" → q3=1,q2=1,q1=0,q0=0 → index 12. ✓
                    for s in selected_states:
                        probs_t[s].append(float(probs_all[int(s, 2)]))

                    progress.progress((ti + 1) / n_pts, text=f"τ = {t:.3f} ...")
                progress.empty()

                # ── Plot ──────────────────────────────────
                palette = [
                    MAROON, "#4E6A9E", "#2E8B57", "#D4853A",
                    "#7B3F9E", "#C0392B", "#1A7A6E", "#8B6914",
                    "#5D6D7E", "#A93226", "#148F77", "#B7950B",
                    "#6E2F8E", "#1A5276", "#117A65", "#784212",
                ]

                fig, ax = plt.subplots(figsize=(9, 4))
                for si, s in enumerate(selected_states):
                    ax.plot(times, probs_t[s], "-o",
                            color=palette[si % len(palette)],
                            ms=3, lw=1.8, label=f"|{s}⟩")

                ax.set_xlabel("Time τ", fontsize=12)
                ax.set_ylabel("Probability", fontsize=12)
                ax.set_ylim(-0.02, 1.05)
                ax.set_xlim(times[0], times[-1])
                ax.set_title(
                    f"Hubbard dynamics  (J={J_val}, U={U_val}, "
                    f"{dyn_steps} Trotter steps)",
                    fontsize=12,
                )
                ax.legend(fontsize=10, loc="upper right",
                          framealpha=0.9, edgecolor="#E0C0C0")
                ax.spines[["top", "right"]].set_visible(False)
                ax.set_facecolor("#FAFAFA")
                fig.patch.set_facecolor("white")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                with st.expander("Raw probability table"):
                    import pandas as pd
                    df = pd.DataFrame({"τ": times,
                                       **{f"|{s}⟩": probs_t[s] for s in selected_states}})
                    st.dataframe(df.style.format("{:.4f}"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2 – CUSTOM CIRCUIT
# ═══════════════════════════════════════════════════════════════
with tab_custom:

    SINGLE_QUBIT_GATES = ["H", "X", "Z", "RX", "RY", "RZ"]
    TWO_QUBIT_GATES    = ["CX", "SWAP"]
    ALL_GATES          = SINGLE_QUBIT_GATES + TWO_QUBIT_GATES

    if "custom_ops" not in st.session_state:
        st.session_state.custom_ops = []

    def op_label(op):
        g = op["gate"]
        if g in ["RX", "RY", "RZ"]:
            return f"{g}(t={op['param']:.3f}) q{op['qubit']}"
        elif g in ["CX", "SWAP"]:
            return f"{g} q{op['control']} -> q{op['target']}"
        else:
            return f"{g} q{op['qubit']}"

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

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown("### Custom Circuit Builder")
        num_qubits = st.slider("Number of qubits", 1, 10, 3, key="custom_nq")
        st.markdown("---")

        st.markdown('<div class="section-label">Add a gate</div>', unsafe_allow_html=True)
        gate = st.selectbox("Gate type", ALL_GATES, key="custom_gate")

        if gate in ["RX", "RY", "RZ"]:
            param = st.slider("Angle theta", 0.0, float(2 * np.pi), 1.0,
                              format="%.3f", key="custom_angle")
        else:
            param = 0.0

        if gate in TWO_QUBIT_GATES:
            label_a = "Control qubit" if gate == "CX" else "Qubit A"
            label_b = "Target qubit"  if gate == "CX" else "Qubit B"
            ca, cb = st.columns(2)
            with ca:
                ctrl = int(st.number_input(label_a, 0, num_qubits - 1, 0, key="custom_ctrl"))
            with cb:
                tgt  = int(st.number_input(label_b, 0, num_qubits - 1,
                                           min(1, num_qubits - 1), key="custom_tgt"))
            qubit = 0
        else:
            qubit = int(st.number_input("Target qubit", 0, num_qubits - 1, 0,
                                        key="custom_qubit"))
            ctrl = tgt = 0

        if st.button("Append Gate", key="add_gate", use_container_width=True):
            st.session_state.custom_ops.append(make_op(gate, qubit, ctrl, tgt, param))
            st.rerun()

        st.markdown("---")
        ops = st.session_state.custom_ops

        if ops:
            st.markdown('<div class="section-label">Gate sequence</div>',
                        unsafe_allow_html=True)
            st.caption("up/dn reorder | del delete | + insert selected gate after this row")

            hc = st.columns([2, 1, 1, 1, 1])
            for col, lbl in zip(hc, ["Gate", "up", "dn", "del", "+"]):
                col.markdown(f"<small><b>{lbl}</b></small>", unsafe_allow_html=True)

            action = None
            for i, op in enumerate(ops):
                rc = st.columns([2, 1, 1, 1, 1])
                rc[0].markdown(
                    f"`{i+1}.` <span style='font-family:monospace;font-size:0.8rem'>"
                    f"{op_label(op)}</span>",
                    unsafe_allow_html=True,
                )
                if rc[1].button("up", key=f"up_{i}", disabled=(i == 0),
                                use_container_width=True):
                    action = ("move_up", i)
                if rc[2].button("dn", key=f"dn_{i}", disabled=(i == len(ops) - 1),
                                use_container_width=True):
                    action = ("move_down", i)
                if rc[3].button("del", key=f"del_{i}", use_container_width=True):
                    action = ("delete", i)
                if rc[4].button("+", key=f"ins_{i}", use_container_width=True):
                    action = ("insert_after", i)

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
            if st.button("Clear All", key="clear_gates", use_container_width=True):
                st.session_state.custom_ops = []
                st.rerun()
        else:
            st.info("Click Append Gate to start building your circuit.")

    with right:
        ops = st.session_state.custom_ops
        if ops:
            circuit = build_custom_circuit(num_qubits, ops)
            draw_circuit(circuit, "Circuit Preview")
            st.markdown("")
            if st.button("Run Simulation", key="run_custom", use_container_width=True):
                with st.spinner("Simulating ..."):
                    counts = run_and_get_histogram(circuit)
                show_histogram(counts, "Measurement Results")
        else:
            st.markdown("""
            <div class="card" style="margin-top:3rem; text-align:center; color:#999;">
              <p style="font-size:1.1rem;">Your circuit will appear here.</p>
              <p>Add gates using the panel on the left.</p>
            </div>
            """, unsafe_allow_html=True)
