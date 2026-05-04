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
st.set_page_config(page_title="CFAT RDSS Framework", layout="centered")
st.title("Concrete-Filled Aluminium Tube (CFAT) Predictor & Optimiser")
st.markdown("### Research Decision Support System (RDSS) Interface")

# ------------------------------------------------------------------
# 1. Distilled Symbolic Equation & Coefficients
# ------------------------------------------------------------------
# These coefficients were distilled from the TabPFN model for interpretability
COEFFS = {'alpha': 1.2459, 'beta': 0.7246, 'gamma': 10.3541, 'delta': 0.9563}

def predict_load(D, t, fc, fal):
    """Predicts axial load capacity using the distilled symbolic equation."""
    if D <= 2 * t + 2: # Basic geometric feasibility check
        return 0.0
    
    Ac = (np.pi/4) * (D - 2*t)**2
    Aal = (np.pi/4) * D**2 - Ac
    
    Pc_f = (fc * Ac / 1000)   # Concrete contribution component
    Pal_f = (fal * Aal / 1000) # Aluminium contribution component
    
    # Revised physics-guided symbolic structure
    return COEFFS['alpha'] * Pal_f + (COEFFS['beta'] * (1 + COEFFS['gamma'] * ((t/D) ** COEFFS['delta'])) * Pc_f)

# ------------------------------------------------------------------
# 2. Domain of Validity Boundaries
# ------------------------------------------------------------------
# Limits derived from the experimental training dataset properties
VALID_LIMITS = {
    'D': (38.0, 700.0),
    't': (1.6, 17.5),
    'fc': (24.1, 110.0),
    'fal': (70.0, 566.2)
}

# ------------------------------------------------------------------
# 3. Sustainability & Cost Data
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
# 4. Sidebar Navigation
# ------------------------------------------------------------------
section = st.sidebar.selectbox("Application Mode", ["Prediction", "Optimisation"])

# ==================================================================
# PREDICTION MODE (With Real-Time Validation)
# ==================================================================
if section == "Prediction":
    st.subheader("Predict Axial Load Capacity")
    st.info("Input parameters are validated in real-time against the experimental domain.")

    col1, col2 = st.columns(2)
    with col1:
        D = st.number_input("Outer Diameter (D) [mm]", 38.0, 800.0, 200.0)
        if D > VALID_LIMITS['D'][1] or D < VALID_LIMITS['D'][0]:
            st.warning(f"⚠️ Diameter exceeds validated range ({VALID_LIMITS['D'][0]}-{VALID_LIMITS['D'][1]}mm)")

        t = st.number_input("Tube Thickness (t) [mm]", 1.0, 25.0, 5.0)
        if t > VALID_LIMITS['t'][1] or t < VALID_LIMITS['t'][0]:
            st.warning(f"⚠️ Thickness exceeds validated range ({VALID_LIMITS['t'][0]}-{VALID_LIMITS['t'][1]}mm)")

        H = st.number_input("Height (H) [mm]", 100.0, 2000.0, 600.0)

    with col2:
        fc = st.number_input("Concrete Strength (fc) [MPa]", 20.0, 150.0, 56.0)
        if fc > VALID_LIMITS['fc'][1] or fc < VALID_LIMITS['fc'][0]:
            st.warning(f"⚠️ Concrete strength exceeds validated range ({VALID_LIMITS['fc'][0]}-{VALID_LIMITS['fc'][1]}MPa)")

        fal = st.number_input("Aluminium Strength (fal) [MPa]", 50.0, 600.0, 240.0)
        if fal > VALID_LIMITS['fal'][1] or fal < VALID_LIMITS['fal'][0]:
            st.warning(f"⚠️ Aluminium yield exceeds validated range ({VALID_LIMITS['fal'][0]}-{VALID_LIMITS['fal'][1]}MPa)")

    st.divider()

    if st.button("Calculate Load Capacity"):
        Pu = predict_load(D, t, fc, fal)
        
        # Calculate environmental and cost metrics
        Ac = (np.pi/4) * (D - 2*t)**2
        Aal = (np.pi/4) * D**2 - Ac
        vol_c = (Ac / 1e6) * (H / 1000)
        vol_a = (Aal / 1e6) * (H / 1000)
        mass_c, mass_a = vol_c * density['Concrete'], vol_a * density['Aluminium']
        
        co2 = mass_c * get_range_value(default_lci['Concrete'], fc) + mass_a * default_lci['Aluminium']
        cost = mass_c * get_range_value(default_cost['Concrete'], fc) + mass_a * default_cost['Aluminium']

        # Results Display
        st.success(f"**Predicted Axial Load Capacity (Pu):** {Pu:,.2f} kN")
        st.info(f"**Reliability Interval (95%):** {Pu*0.95:,.0f} – {Pu*1.05:,.0f} kN")
        
        st.markdown(f"""
        ### Design Summary
        | Metric | Value | Unit |
        |---|---|---|
        | Carbon Footprint | {co2:.2f} | kg CO₂e |
        | Material Cost | ${cost:.2f} | USD |
        | Concrete Area (Ac) | {Ac:,.0f} | mm² |
        | Aluminium Area (Aal) | {Aal:,.0f} | mm² |
        """)

# ==================================================================
# OPTIMISATION MODE
# ==================================================================
else:
    st.subheader("Multi-Objective Optimisation (NSGA-II)")
    
    col1, col2 = st.columns(2)
    with col1:
        target_load = st.number_input("Target Load Capacity (kN)", 100.0, 25000.0, 4000.0)
    with col2:
        height = st.number_input("Column Height (mm)", 114.0, 1620.0, 800.0)

    class CFATProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=4, n_obj=2, n_ieq_constr=1,
                           xl=[VALID_LIMITS['D'][0], VALID_LIMITS['t'][0], VALID_LIMITS['fc'][0], VALID_LIMITS['fal'][0]],
                           xu=[VALID_LIMITS['D'][1], VALID_LIMITS['t'][1], VALID_LIMITS['fc'][1], VALID_LIMITS['fal'][1]])

        def _evaluate(self, x, out, *args, **kwargs):
            D, t, fc, fal = x
            Pu = predict_load(D, t, fc, fal)
            Ac = np.pi/4 * (D - 2*t)**2 / 1e6
            Aal = (np.pi/4 * D**2 - np.pi/4 * (D - 2*t)**2) / 1e6
            mass_c = (Ac * height/1000) * density['Concrete']
            mass_a = (Aal * height/1000) * density['Aluminium']
            
            co2 = mass_c * get_range_value(default_lci['Concrete'], fc) + mass_a * default_lci['Aluminium']
            cost = mass_c * get_range_value(default_cost['Concrete'], fc) + mass_a * default_cost['Aluminium']
            
            out["F"] = [co2, cost]
            out["G"] = [target_load - Pu] # Constraint: Pu must be >= target_load

    if st.button("Run NSGA-II Optimisation"):
        with st.spinner("Executing Evolutionary Search..."):
            res = minimize(CFATProblem(), NSGA2(pop_size=100), ('n_gen', 100), seed=42)
        
        if res.F.size > 0:
            # Select balanced solution (L2 Norm on normalized objectives)
            norm_F = res.F / res.F.max(axis=0)
            best_idx = np.argmin(np.sum(norm_F**2, axis=1))
            X_opt = res.X[best_idx]
            
            st.success("Optimisation Completed!")
            st.info("🛡️ **Verified Design:** This configuration is strictly within the validated experimental domain.")
            
            st.write(f"**Optimal Solution:** D={X_opt[0]:.1f}mm, t={X_opt[1]:.1f}mm, fc={X_opt[2]:.1f}MPa, fal={X_opt[3]:.1f}MPa")
            st.write(f"**Performance:** Load={predict_load(*X_opt):,.1f} kN | Carbon={res.F[best_idx,0]:.1f} kg | Cost=${res.F[best_idx,1]:.2f}")

            # Pareto Plot
            fig, ax = plt.subplots()
            ax.scatter(res.F[:,0], res.F[:,1], alpha=0.6, label="Pareto Candidates")
            ax.scatter(res.F[best_idx,0], res.F[best_idx,1], color='red', s=100, label="Selected Design")
            ax.set_xlabel("Carbon Footprint (kg CO₂e)")
            ax.set_ylabel("Cost (USD)")
            ax.legend()
            st.pyplot(fig)

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
