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
    
    # Corrected Geometry: Explicit hollow tube formula
    Ac = (np.pi/4) * (D - 2*t)**2
    Aal = (np.pi/4) * (D**2 - (D - 2*t)**2)
    
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
        
        # Consistent Geometry Logic
        Ac = (np.pi/4) * (D - 2*t)**2
        Aal = (np.pi/4) * (D**2 - (D - 2*t)**2)

        # Baseline volume and mass tracking based on a 1-meter (1000mm) profile
        vol_c_per_m = Ac / 1e6 * 1.0
        vol_a_per_m = Aal / 1e6 * 1.0
        mass_c_per_m = vol_c_per_m * density['Concrete']
        mass_a_per_m = vol_a_per_m * density['Aluminium']

        # Unit length profiles
        co2_per_m  = mass_c_per_m * get_range_value(default_lci['Concrete'], fc)  + mass_a_per_m * default_lci['Aluminium']
        cost_per_m = mass_c_per_m * get_range_value(default_cost['Concrete'], fc) + mass_a_per_m * default_cost['Aluminium']

        # Total scaling using structural parameter multipliers (Height converted to meters)
        height_m = H / 1000.0
        total_co2 = co2_per_m * height_m
        total_cost = cost_per_m * height_m

        st.success(f"Predicted Axial Load Capacity: {Pu:,.2f} kN")

        st.markdown(f"""
        ### Design Summary
        | Parameter                      | Value           | Unit    |
        |-------------------------------|-----------------|---------|
        | Diameter (D)                  | {D:.2f}         | mm      |
        | Thickness (t)                 | {t:.2f}         | mm      |
        | **Concrete Area (Ac)** | {Ac:,.0f}       | mm²     |
        | **Aluminium Area (Aal)** | {Aal:,.0f}      | mm²     |
        | Concrete strength (fc)        | {fc:.1f}        | MPa     |
        | Aluminium strength (fal)      | {fal:.1f}       | MPa     |
        | Height                        | {H:.0f}         | mm      |
        | **Unit-length Carbon Footprint**| {co2_per_m:.2f}  | kg CO₂e/m |
        | **Unit-length Material Cost** | ${cost_per_m:.2f} | USD/m   |
        | **Total Carbon Footprint** | {total_co2:.2f}  | kg CO₂e |
        | **Total Material Cost** | ${total_cost:.2f}| USD     |
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
        max_cost   = st.number_input("Maximum Allowed Total Cost (USD)", 0.0, 20000.0, 1000.0)
        max_carbon = st.number_input("Maximum Allowed Total Carbon (kg CO₂e)", 0.0, 20000.0, 1500.0)

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
            
            # Cross-sectional logic mapped into m²
            Ac = (np.pi/4) * (D - 2*t)**2 / 1e6
            Aal = (np.pi/4 * (D**2 - (D - 2*t)**2)) / 1e6
            
            # Mass profile of a unit 1-meter structural volume
            vol_c_per_m, vol_a_per_m = Ac * 1.0, Aal * 1.0
            mass_c_per_m, mass_a_per_m = vol_c_per_m * density['Concrete'], vol_a_per_m * density['Aluminium']
            
            co2_per_m  = mass_c_per_m * get_range_value(c_lci, fc)  + mass_a_per_m * al_lci
            cost_per_m = mass_c_per_m * get_range_value(c_cost, fc) + mass_a_per_m * al_cost
            
            # Multiply unit values by column length parameter (converted to meters)
            height_m = height / 1000.0
            total_co2 = co2_per_m * height_m
            total_cost = cost_per_m * height_m
            
            out["F"] = [total_co2, total_cost]
            out["G"] = [
                target_load - Pu,
                total_co2 - max_carbon,
                total_cost - max_cost
            ]

    if st.button("Run NSGA-II Optimisation"):
        with st.spinner("Running NSGA-II..."):
            res = minimize(CFATProblem(), NSGA2(pop_size=120), ('n_gen', 120), seed=42)

        if res.F.size > 0:
            F = res.F
            norm = F / F.max(axis=0)
            best_idx = np.argmin(np.sum(norm**2, axis=1))
            D_opt, t_opt, fc_opt, fal_opt = res.X[best_idx]

            # Re-calculating components for performance breakdown display
            Pu_opt = predict_load(D_opt, t_opt, fc_opt, fal_opt)
            Ac_opt  = (np.pi/4) * (D_opt - 2*t_opt)**2
            Aal_opt = (np.pi/4) * (D_opt**2 - (D_opt - 2*t_opt)**2)
            
            vol_c_per_m_opt = Ac_opt / 1e6 * 1.0
            vol_a_per_m_opt = Aal_opt / 1e6 * 1.0
            mass_c_per_m_opt = vol_c_per_m_opt * density['Concrete']
            mass_a_per_m_opt = vol_a_per_m_opt * density['Aluminium']

            carbon_per_m_opt = mass_c_per_m_opt * get_range_value(c_lci, fc_opt) + mass_a_per_m_opt * al_lci
            cost_per_m_opt   = mass_c_per_m_opt * get_range_value(c_cost, fc_opt) + mass_a_per_m_opt * al_cost

            # Total optimization profile length multiplication
            height_m = height / 1000.0
            total_carbon_opt = carbon_per_m_opt * height_m
            total_cost_opt   = cost_per_m_opt * height_m

            st.success("Optimisation Completed Successfully!")
            st.markdown(f"""
            ### Optimal CFAT Design
            | Parameter                         | Value               | Unit    |
            |----------------------------------|-------------------|---------|
            | **Diameter (D)** | {D_opt:.2f}       | mm      |
            | **Thickness (t)** | {t_opt:.2f}       | mm      |
            | **Concrete Area (Ac)** | {Ac_opt:,.0f}     | mm²     |
            | **Aluminium Area (Aal)** | {Aal_opt:,.0f}    | mm²     |
            | **Concrete strength (fc)** | {fc_opt:.2f}      | MPa     |
            | **Aluminium strength (fal)** | {fal_opt:.1f}     | MPa     |
            | **Height** | {height:.0f}      | mm      |
            | **Predicted Load Capacity** | {Pu_opt:,.2f}     | kN      |
            | **Unit-length Carbon Footprint** | {carbon_per_m_opt:.2f} | kg CO₂e/m |
            | **Unit-length Material Cost** | ${cost_per_m_opt:.2f}  | USD/m   |
            | **Total Carbon Footprint** | {total_carbon_opt:.2f} | kg CO₂e |
            | **Total Material Cost** | ${total_cost_opt:.2f}  | USD     |
            """)

            fig, ax = plt.subplots(figsize=(7.5, 5))
            ax.scatter(F[:,0], F[:,1], c='lightblue', edgecolor='navy', alpha=0.7, s=60)
            ax.scatter(total_carbon_opt, total_cost_opt, c='red', s=200, label='Selected solution', zorder=5)
            ax.set_xlabel("Total Carbon Footprint (kg CO₂e)")
            ax.set_ylabel("Total Material Cost (USD)")
            ax.set_title("Pareto Front – Total Carbon vs Total Cost")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.error("No feasible solution found.")

st.divider()
st.markdown("""
    **Notes**: This application predicts the axial load capacity of CFAT columns using a distilled symbolic equation from TabPFN.
""")

footer_html = """
<style>.footer { position: fixed; left: 0; bottom: 0; width: 100%; background-color: #f1f1f1; text-align: center; padding: 10px; font-size: 12px; color: #6c757d; }</style>
<div class="footer"><p>© 2026 | Temitope E. Dada et al. | For Queries: <a href="mailto:temitope.dada@teaktecheng.com">temitope.dada@teaktecheng.com</a></p></div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
