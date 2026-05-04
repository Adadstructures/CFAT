import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
import warnings

# Suppress warnings for a clean UI
warnings.filterwarnings("ignore")

# --- App Configuration ---
st.set_page_config(page_title="CFAT Predictor & Optimiser", layout="centered")
st.title("Concrete-Filled Aluminium Tube (CFAT) Predictor & Optimiser")
st.markdown("### Research Decision Support System (RDSS) Interface")

# ------------------------------------------------------------------
# 1. Distilled Symbolic Equation & Coefficients
# ------------------------------------------------------------------
# Updated coefficients based on the latest research distillation
COEFFS = {'alpha': 1.2459, 'beta': 0.7246, 'gamma': 10.3541, 'delta': 0.9563}

def predict_load(D, t, fc, fal):
    """Predicts axial load capacity using the distilled symbolic equation."""
    if D <= 2 * t + 5: # Basic geometric feasibility check
        return 0.0
    
    Ac = (np.pi/4) * (D - 2*t)**2
    Aal = (np.pi/4) * D**2 - Ac
    
    Pc_f = (fc * Ac / 1000)   # Concrete contribution component
    Pal_f = (fal * Aal / 1000) # Aluminium contribution component
    
    # Physics-guided symbolic structure
    return COEFFS['alpha'] * Pal_f + (COEFFS['beta'] * (1 + COEFFS['gamma'] * ((t/D) ** COEFFS['delta'])) * Pc_f)

# ------------------------------------------------------------------
# 2. Material data & LCI
# ------------------------------------------------------------------
default_lci = {
    'Aluminium': 8.78,
    'Concrete': {(21,30):0.135, (31,59):0.180, (60,99):0.225, (100,131):0.325}
}

default_cost = {
    'Aluminium': 4.5,
    'Concrete': {(21,30):0.125, (31,59):0.175, (60,99):0.225, (100,131):0.30}
}

density = {'Aluminium': 2700, 'Concrete': 2400}

def get_range_value(dct, fc):
    for (lo, hi), val in dct.items():
        if lo <= fc <= hi:
            return val
    return list(dct.values())[-1]

# ------------------------------------------------------------------
# 3. Sidebar Navigation
# ------------------------------------------------------------------
section = st.sidebar.selectbox("Mode", ["Prediction", "Optimisation"])

# ==================================================================
# PREDICTION MODE
# ==================================================================
if section == "Prediction":
    st.subheader("Predict Axial Load Capacity")

    col1, col2 = st.columns(2)
    with col1:
        D  = st.number_input("Outer Diameter (mm)", 38.0, 700.0, 200.0)
        t  = st.number_input("Tube Thickness (mm)", 1.6, 17.5, 5.0)
        H  = st.number_input("Height (mm)", 114.0, 1620.0, 600.0)
    with col2:
        fc  = st.number_input("Concrete Strength fc (MPa)", 24.1, 110.0, 56.0)
        fal = st.number_input("Aluminium Yield Strength fal (MPa)", 70.0, 566.2, 240.0)

    if st.button("Calculate Load Capacity"):
        Pu = predict_load(D, t, fc, fal)
        Ac  = np.pi/4 * (D - 2*t)**2
        Aal = np.pi/4 * (D**2 - Ac)

        vol_c = Ac / 1e6 * (H/1000)
        vol_a = Aal / 1e6 * (H/1000)
        mass_c = vol_c * density['Concrete']
        mass_a = vol_a * density['Aluminium']

        co2  = mass_c * get_range_value(default_lci['Concrete'], fc)  + mass_a * default_lci['Aluminium']
        cost = mass_c * get_range_value(default_cost['Concrete'], fc) + mass_a * default_cost['Aluminium']

        st.success(f"Predicted Axial Load Capacity: {Pu:,.2f} kN")
        st.write(f"95% approximate interval: **{Pu*0.95:,.0f} – {Pu*1.05:,.0f} kN**")

        st.markdown(f"""
        ### Design Summary
        | Parameter                     | Value           | Unit    |
        |-------------------------------|-----------------|---------|
        | Diameter (D)                  | {D:.2f}         | mm      |
        | Thickness (t)                 | {t:.2f}         | mm      |
        | **Concrete Area (Ac)**        | {Ac:,.0f}       | mm²     |
        | **Aluminium Area (Aal)**      | {Aal:,.0f}      | mm²     |
        | Concrete strength (fc)        | {fc:.1f}        | MPa     |
        | Aluminium strength (fal)      | {fal:.1f}       | MPa     |
        | Height                        | {H:.0f}         | mm      |
        | Carbon footprint              | {co2:.2f}       | kg CO₂e |
        | Material cost                 | ${cost:.2f}     | USD     |
        """)

# ==================================================================
# OPTIMISATION MODE
# ==================================================================
else:
    st.subheader("Multi-Objective Optimisation (NSGA-II)")

    col1, col2 = st.columns(2)
    with col1:
        target_load = st.number_input("Target Load Capacity (kN)", 80.0, 25000.0, 4000.0)
        height = st.number_input("Column Height (mm)", 114.0, 1620.0, 800.0)
    with col2:
        max_cost   = st.number_input("Maximum Allowed Cost (USD)", 0.0, 20000.0, 1000.0)
        max_carbon = st.number_input("Maximum Allowed Carbon (kg CO₂e)", 0.0, 20000.0, 1500.0)

    use_custom = st.checkbox("Use custom LCI / Cost values")

    if use_custom:
        al_lci  = st.number_input("Aluminium LCI (kg CO₂e/kg)", value=8.78)
        al_cost = st.number_input("Aluminium Cost (USD/kg)", value=4.5)
        st.markdown("**Concrete — banded values**")
        c_lci = {
            (21,30):   st.number_input("Concrete LCI 21–30 MPa",   value=0.135),
            (31,59):   st.number_input("Concrete LCI 31–59 MPa",   value=0.180),
            (60,99):   st.number_input("Concrete LCI 60–99 MPa",   value=0.225),
            (100,131): st.number_input("Concrete LCI 100–131 MPa", value=0.325)
        }
        c_cost = {
            (21,30):   st.number_input("Concrete Cost 21–30 MPa",   value=0.125),
            (31,59):   st.number_input("Concrete Cost 31–59 MPa",   value=0.175),
            (60,99):   st.number_input("Concrete Cost 60–99 MPa",   value=0.225),
            (100,131): st.number_input("Concrete Cost 100–131 MPa", value=0.30)
        }
    else:
        al_lci, al_cost = default_lci['Aluminium'], default_cost['Aluminium']
        c_lci, c_cost   = default_lci['Concrete'],  default_cost['Concrete']

    # NSGA-II Problem Definition
    class CFATProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=4, n_obj=2, n_ieq_constr=3,
                           xl=[38.0, 1.6, 24.1, 70.0],
                           xu=[700.0, 17.5, 110.0, 566.2])

        def _evaluate(self, x, out, *args, **kwargs):
            D, t, fc, fal = x
            Pu = predict_load(D, t, fc, fal)
            Ac  = np.pi/4 * (D - 2*t)**2 / 1e6
            Aal = (np.pi/4 * D**2 - np.pi/4 * (D - 2*t)**2) / 1e6
            vol_c, vol_a = Ac * (height/1000), Aal * (height/1000)
            mass_c, mass_a = vol_c * density['Concrete'], vol_a * density['Aluminium']
            
            co2  = mass_c * get_range_value(c_lci, fc)  + mass_a * al_lci
            cost = mass_c * get_range_value(c_cost, fc) + mass_a * al_cost
            
            out["F"] = [co2, cost]
            # Constraints: Target Load, Max Carbon, Max Cost
            out["G"] = [
                target_load - Pu,
                co2 - max_carbon,
                cost - max_cost
            ]

    if st.button("Run NSGA-II Optimisation"):
        with st.spinner("Running NSGA-II (120 individuals × 120 generations)..."):
            res = minimize(CFATProblem(), NSGA2(pop_size=120), ('n_gen', 120), seed=42)

        if res.F.size > 0:
            F = res.F
            norm = F / F.max(axis=0)
            best_idx = np.argmin(np.sum(norm**2, axis=1))
            D_opt, t_opt, fc_opt, fal_opt = res.X[best_idx]

            Pu_opt = predict_load(D_opt, t_opt, fc_opt, fal_opt)
            Ac_opt  = np.pi/4 * (D_opt - 2*t_opt)**2
            Aal_opt = np.pi/4 * (D_opt**2 - Ac_opt)
            vol_c_opt = Ac_opt / 1e6 * (height/1000)
            vol_a_opt = Aal_opt / 1e6 * (height/1000)
            mass_c_opt, mass_a_opt = vol_c_opt * density['Concrete'], vol_a_opt * density['Aluminium']

            carbon_opt = mass_c_opt * get_range_value(c_lci, fc_opt) + mass_a_opt * al_lci
            cost_opt   = mass_c_opt * get_range_value(c_cost, fc_opt) + mass_a_opt * al_cost

            st.success("Optimisation Completed Successfully!")
            st.markdown(f"""
            ### Optimal CFAT Design
            | Parameter                        | Value             | Unit    |
            |----------------------------------|-------------------|---------|
            | **Diameter (D)**                 | {D_opt:.2f}       | mm      |
            | **Thickness (t)**                | {t_opt:.2f}       | mm      |
            | **Concrete Area (Ac)**           | {Ac_opt:,.0f}     | mm²     |
            | **Aluminium Area (Aal)**         | {Aal_opt:,.0f}    | mm²     |
            | **Concrete strength (fc)**       | {fc_opt:.2f}      | MPa     |
            | **Aluminium strength (fal)**     | {fal_opt:.1f}     | MPa     |
            | **Height**                       | {height:.0f}      | mm      |
            | **Predicted Load Capacity**      | {Pu_opt:,.0f}     | kN      |
            | **Total Carbon Footprint**       | {carbon_opt:.1f}  | kg CO₂e |
            | **Total Material Cost**          | ${cost_opt:.2f}   | USD     |
            """)

            # Pareto Plot
            fig, ax = plt.subplots(figsize=(7.5, 5))
            ax.scatter(F[:,0], F[:,1], c='lightblue', edgecolor='navy', alpha=0.7, s=60)
            ax.scatter(carbon_opt, cost_opt, c='red', s=200, label='Selected solution', zorder=5)
            ax.set_xlabel("Carbon Footprint (kg CO₂e)")
            ax.set_ylabel("Material Cost (USD)")
            ax.set_title("Pareto Front – Carbon vs Cost")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.error("No feasible solution found. Try increasing target tolerances or constraints.")

# ------------------------------------------------------------------
# Research Footer & RDSS Context
# ------------------------------------------------------------------
st.divider()
st.markdown("""
    **Notes**: 
    1. This application predicts the axial load capacity of concrete filled aluminium tube (CFAT) columns using a distilled symbolic equation from TabPFN.  
    2. The model was trained on a curated dataset of experimental specimens with varying material strengths and configurations.  
    3. Multi-objective optimisation is incorporated to recommend sustainable and cost-effective design parameters that meet target structural performance.  
""")

st.markdown("""
    **References**: 
    1. Deb K, Pratap A, Agarwal S, Meyarivan T. A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE Trans Evol Computat 2002;6:182–97. https://doi.org/10.1109/4235.996017.
    2. Hollmann N, Müller S, Eggensperger K, Hutter F. TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second 2022. https://doi.org/10.48550/ARXIV.2207.01848.
""")

footer_html = """
<style>
.footer { position: fixed; left: 0; bottom: 0; width: 100%; background-color: #f1f1f1; text-align: center; padding: 10px; font-size: 12px; color: #6c757d; }
</style>
<div class="footer">
    <p>© 2026 | Temitope E. Dada, Silas E. Oluwadahunsi, Oluwafemi Omotayo, Charles K.S. Moy and Yao Sun | For Queries: <a href="mailto:temitope.dada@teaktecheng.com">temitope.dada@teaktecheng.com</a></p>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
