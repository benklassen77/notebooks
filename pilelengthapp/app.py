import numpy as np
from scipy.optimize import dual_annealing
from scipy.integrate import trapezoid
from sklearn.metrics import max_error
import matplotlib.pyplot as plt
import streamlit as st

# Set page layout
st.set_page_config(layout="wide")

# Constants (editable if desired)
E = 35000 # MPa
I = ((750**4)/12)/10 # mm^4
diameter = 0.75 # m
P = st.sidebar.slider("Point Load P (kN)", 100, 500, 270)
d1 = st.sidebar.slider("Depth of Point Load Above Ground d1 (m)", 1.0, 5.0, 1.8)
c_u = st.sidebar.slider("Undrained Shear Strength cu (kPa)", 20, 100, 50)
e_50 = 0.007
y_50 = 2.5 * e_50 * diameter
J = 0.5
gamma = 21
gamma_p = gamma - 9.81

def calc_pu(z):
    return min(3 + (gamma_p / c_u + J / diameter) * z, 9) * c_u * diameter

def calc_p(p_u, y):
    p = p_u / 2 * (abs(y / 1000) / y_50)**(1/3)
    if y > 0:
        p = -p
    return p * 0.9

bounds = [(0.1, 25), (0.1, 25), (0.001, np.deg2rad(3.5))]
discretization = 100

@st.cache_data(show_spinner=True)
def run_model(P, d1, c_u):
    def f(vars):
        d2, d3, theta = vars
        l = d1 + d2 + d3
        dx = l / discretization * 1000
        x = np.linspace(0, l, discretization + 1)
        x_ref_d1 = (x - d1) * (x >= d1)
        x_ref_d2 = -1 * (x - d1 - d2)
        pu_i = [calc_pu(z) for z in x_ref_d1]

        M_cant = P * x
        y_cant = P * 1000 / (6 * E * I) * (2 * (l * 1000)**3 - 3 * (l * 1000)**2 * (x * 1000) + (x * 1000)**3)
        y_rot = x_ref_d2 * theta * 1000

        y_prev = y_cant + y_rot
        y_soil_prev = y_prev * (x > d1)
        tolerance = 0.5
        diff = 1000
        iteration = 0

        while diff > tolerance:
            M_soil = np.zeros(len(x))
            P_soil = np.zeros(len(x))
            M = np.zeros(len(x))
            y = np.zeros(len(x))
            slope = np.zeros(len(x))

            for i in range(len(x)):
                P_soil[i] = calc_p(pu_i[i], y_soil_prev[i])

            for i in range(len(x)):
                lever_arm = (x[i] - x) * (x <= x[i])
                M_soil[i] = trapezoid(P_soil * lever_arm, x)

            M = M_cant + M_soil
            M_over_EI = np.flip(M) / (E * I) * 1000**2
            slope[1:] = np.cumsum(0.5 * (M_over_EI[1:] + M_over_EI[:-1]) * dx)
            y[1:] = np.cumsum(0.5 * (slope[1:] + slope[:-1]) * dx)
            y = np.flip(y) + y_rot
            diff = max_error(y, y_prev)
            y_prev = y
            y_soil_prev = y_prev * (x > d1)
            iteration += 1
            if iteration > 5:
                break

        M_bot = M[-1]
        P_soil_flipped = np.flip(P_soil)
        x_flipped = np.flip(x)
        M_top = trapezoid(P_soil_flipped * x_flipped, x)
        Sum_F = trapezoid(P_soil, x) + P
        objective = M_bot**2 + M_top**2 + Sum_F**2

        return objective, x, M, y, d2, d3, np.rad2deg(theta), l, P_soil

    def calc_obj(vars):
        return f(vars)[0]

    result = dual_annealing(calc_obj, bounds, seed=42)
    return f(result.x)

# Run simulation
with st.spinner("Optimizing..."):
    val, x, M, y, d2, d3, theta_deg, l, P_soil = run_model(P, d1, c_u)

# Results
st.success("Optimization Complete")
st.write(f"**Objective Function Value:** {val:.2f}")
st.write(f"**Optimal d2:** {d2:.2f} m")
st.write(f"**Optimal d3:** {d3:.2f} m")
st.write(f"**Optimal rotation θ:** {theta_deg:.3f}°")
st.write(f"**Total Embedment:** {d2 + d3:.2f} m")

# Plotting
fig1, ax1 = plt.subplots()
ax1.plot(M, x)
ax1.axhline(d1, color='k', linestyle='--')
ax1.set_xlabel("Moment (kNm)")
ax1.set_ylabel("Depth (m)")
ax1.set_title("Moment Distribution")
ax1.invert_yaxis()
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
ax2.plot(y, x)
ax2.axhline(d1, color='k', linestyle='--')
ax2.set_xlabel("Deflection (mm)")
ax2.set_ylabel("Depth (m)")
ax2.set_title("Deflected Shape")
ax2.invert_yaxis()
st.pyplot(fig2)

fig3, ax3 = plt.subplots()
ax3.plot(-P_soil, x)
ax3.axhline(d1, color='k', linestyle='--')
ax3.set_xlabel("Soil Pressure (kN/m)")
ax3.set_ylabel("Depth (m)")
ax3.set_title("Soil Pressure Distribution")
ax3.invert_yaxis()
st.pyplot(fig3)
