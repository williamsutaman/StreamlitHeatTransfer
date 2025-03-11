'''
PETR-5071 DRILLING AND DESIGN FOR GEOTHERMAL
Project 2: Heat Transfer Simulator

Williams Utaman
900365203
Spring Semester 2025
'''

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

#STREAMLIT END USER INTERFACE - INTRODUCTION

#Setting up the main Streamlit interface
st.title("Heat Transfer Simulator for Simple Oil and Gas Wells")
st.sidebar.header("Simulation Settings")
#Selecting between drilling and production cases
WellType = st.sidebar.radio("Select your well type here:", ("Drilling Well Case", "Production Well Case"))

#Defining all the units according to the unit system
ConversionFactors = {
    "TVD": 0.3048,                              #ft to m
    "q_ProdFluid": 0.158987,                    #STBD to m³/d
    "SG_Gas": 1,                                #Dimensionless
    "SG_Oil": 1,                                #Dimensionless
    "rho_AnnFluid": 16.018449,                  #lbm/ft³ to kg/m³
    "mu_AnnFluid": 0.995146,                    #cP to Pa-s
    "mu_Gas": 0.995146,                         #cP to Pa-s
    "GOR": 0.178108,                            #SCF/STB to SM³/SM³
    "TubingID": 0.0254,                         #in to m
    "TubingOD": 0.0254,                         #in to m
    "CasingID": 0.0254,                         #in to m
    "CasingOD": 0.0254,                         #in to m
    "WellboreDiameter": 0.0254,                 #in to m
    "P_Wellhead": 6.89476,                      #psi to kPa
    "T_Surface": lambda x: (x - 32) * 5 / 9,    #°F to °C
    "T_Bottomhole": lambda x: (x - 32) * 5 / 9, #°F to °C
    "t_Production": 1,                          #No change
    "k_Earth": 0.0720657,                       #BTU/D-ft-°F to W/m-K
    "k_Cement": 0.0720657,                      #BTU/D-ft-°F to W/m-K
    "k_Casing": 0.0720657,                      #BTU/D-ft-°F to W/m-K
    "k_Tubing": 0.0720657,                      #BTU/D-ft-°F to W/m-K
    "k_AnnFluid": 0.0720657,                    #BTU/D-ft-°F to W/m-K
    "k_ProdFluid": 0.0720657,                   #BTU/D-ft-°F to W/m-K
    "Cp_AnnFluid": 4.1868,                      #BTU/lbm-°F to kJ/kg-K
    "Cp_ProdFluid": 4.1868,                     #BTU/lbm-°F to kJ/kg-K
    "rho_Earth": 16.018449,                     #lbm/ft³ to kg/m³
    "Cp_Earth": 4.1868,                         #BTU/lbm-°F to kJ/kg-K
}

#Defining all the units according to the oil field units
Units = {
    "TVD": "ft",
    "q_ProdFluid": "STBD",
    "SG_Gas": "dimensionless",
    "SG_Oil": "dimensionless",
    "rho_AnnFluid": "lbm/ft³",
    "mu_AnnFluid": "cP",
    "mu_ProdFluid": "cP",
    "mu_Gas": "cP",
    "GOR": "SCF/STB",
    "TubingID": "in",
    "TubingOD": "in",
    "CasingID": "in",
    "CasingOD": "in",
    "WellboreDiameter": "in",
    "P_Wellhead": "psi",
    "T_Surface": "°F",
    "T_Bottomhole": "°F",
    "t_Production": "Days",
    "k_Earth": "BTU/D-ft-°F",
    "k_Cement": "BTU/D-ft-°F",
    "k_Casing": "BTU/D-ft-°F",
    "k_Tubing": "BTU/D-ft-°F",
    "k_AnnFluid": "BTU/D-ft-°F",
    "k_ProdFluid": "BTU/D-ft-°F", 
    "Cp_AnnFluid": "BTU/lbm-°F",
    "Cp_ProdFluid": "BTU/lbm-°F",
    "rho_Earth": "lbm/ft³",
    "Cp_Earth": "BTU/lbm-°F",
    }

#HEAT TRANSFER LIBRARY

#Dimensionless Numbers
#Prandtl number(dimensionless)
def Function__N_Pr (Cp, mu, k):
    return 58.1 * Cp * mu / k
#Reynolds number (dimensionless)
def Function__N_Re (rho, u, D, mu):
    return 0.0172 * rho * u * D / mu
#Nusselt number (dimensionless)
def Function__N_Nu (N_Re, N_Pr):
    return 0.023 * (N_Re ** 0.8) * (N_Pr ** 0.3)
#Grashof number (dimensionless)
def Function__N_Gr (r_ci, r_to, rho_ann, beta, mu_ann, delta_T):
    g = 32.1740 * (3600 ** 2) #in ft/hr²
    return 71200000 * ((r_ci - r_to) ** 3 * (rho_ann ** 2) * beta * g * delta_T) / (mu_ann ** 2)

#Heat Transfer Coefficients
#Annular convective heat transfer coefficient --- Natural convection (BTU/D-ft²-°F)
def Function__h_ann (N_Gr, N_Pr, k_ann, r_to, r_ci):
    return (0.049 * ((N_Gr * N_Pr) ** 0.33) * (N_Pr ** 0.074) * k_ann) / (r_to * math.log(r_ci / r_to))
#Production fluid heat transfer coefficient --- Forced convection (BTU/D-ft²-°F)
def Function__h_f (N_Nu, k, d):
    return N_Nu * k / d

#Overall Heat Transfer
#Overall heat transfer coefficient (BTU/ft²-D-°F)
def Function__U (r_to, r_ti, h_f, k_t, h_ann, r_ci, k_c, r_co, k_cem, r_w, k_e, f_t):
    R_conv_f = 1 / (r_ti * h_f)
    R_cond_t = np.log(r_to / r_ti) / k_t
    R_conv_ann = 1 / (r_ci * h_ann)
    R_cond_c = np.log(r_co / r_ci) / k_c
    R_cond_cem = np.log(r_w / r_co) / k_cem
    R_cond_e = f_t / k_e
    return 1 / r_to * (1 / (R_conv_f + R_cond_t + R_conv_ann + R_cond_c + R_cond_cem + R_cond_e))
#Overall heat transfer rate (BTU/D)
def Function__q (r, L, U, T_Fluid, T_Surface, DepthNow, GeothermalGradient):
    T_Surrounding = T_Surface + (GeothermalGradient * DepthNow)
    return 2 * math.pi * r * L * U * (T_Fluid - T_Surrounding)

#Earth Thermal Resistance
#Heat dissipation based on dimensionless transient time (dimensionless)
def Function__f_t (t_Dw):
    if t_Dw <= 1.5:
        return 1.1281 * np.sqrt(t_Dw) * (1 - 0.3 * np.sqrt(t_Dw))
    else:
        return (0.4063 + 0.5 * np.log(t_Dw)) * (1 + 0.6 / t_Dw)
#Dimensionless transient time (dimensionless)
def Function__t_Dw (k_e, t, rho_e, Cp_e, r_w):
    return (k_e * t) / (rho_e * Cp_e * r_w ** 2)

#Parameters to Compensate Kinetic Energy and Joule-Thompson Effect
#Parameter of Fc
def Function__Fc (P_wh, m_dot, GOR, SG_Oil, SG_Gas, T_Bottomhole, T_Surface, TVD):
    API = (141.5 / SG_Oil) - 131.5
    GeothermalGradient = (T_Bottomhole - T_Surface) / TVD
    return (-2.978e-3 + 1.006e-6 * P_wh + 1.906e-4 * (m_dot / 86400) - 1.047e-6 * GOR + 3.229e-5 * API + 4.009e-3 * SG_Gas - 0.3551 * GeothermalGradient)
#Parameter of potential
def Function__Potential (Cp):
    return 1 / (778.169 * Cp)

#Average Temperature
#Parameter of T_f average (°F)
def Function__T_f_average (T_f_i, Q_ann, m_dot, Cp, Fc, Potential, dL):
    return T_f_i - (Q_ann / (2 * m_dot * Cp)) + (Fc - Potential) * dL

#Other Supporting Functions
#Heat transfer through annulus
def Function__Q_ann (r_ci, L, h_ann, delta_T):
    R_ann = 1 / (r_ci * h_ann)
    return 2 * math.pi * R_ann * L * h_ann * delta_T
#Mass flow rate (lbm/D)
def Function__m_dot (q, rho_fluid):
    return 5.615 * q * rho_fluid
#Fluid density calculator (lb/ft³)
def Function__rho_ProdFluid (GOR, SG_Oil, SG_Gas):
    WaterDensity = 62.4         # lb/ft³ at standard conditions
    AirDensity = 0.0765         # lb/ft³ at standard conditions
    OilBarrelVolume = 5.615     # STB to standard ft³; Gas volume is assumed as "GOR" SCF in every STB of oil
    OilMass = (SG_Oil * WaterDensity) * OilBarrelVolume     #Oil mass in lb unit
    GasMass = (SG_Gas * AirDensity) * GOR                   #Gas mass in lb unit
    ProdFluidDensity = (OilMass + GasMass) / (GOR + OilBarrelVolume)
    return ProdFluidDensity

def main_ProductionMultiphase(TVD, q_ProdFluid, CasingID, CasingOD, TubingID, TubingOD, WellboreDiameter, P_Wellhead, T_Surface, T_Bottomhole, t_Production, SG_Oil, SG_Gas, rho_AnnFluid,
                             mu_AnnFluid, mu_ProdFluid, mu_Gas, GOR, k_Earth, k_Cement, k_Casing, k_Tubing, k_AnnFluid, k_ProdFluid, Cp_AnnFluid, Cp_ProdFluid, rho_Earth, Cp_Earth):
    #General Well Data
    TVD = TVD                                       #ft
    q_ProdFluid = q_ProdFluid * 0.158987 / 24       #STB/day to ft³/hr
    CasingID = CasingID / 12 / 2                    #in to ft, then to radius
    CasingOD = CasingOD  / 12 / 2                   #in to ft, then to radius
    TubingID = TubingID / 12 / 2                    #in to ft, then to radius
    TubingOD = TubingOD / 12 / 2                    #in to ft, then to radius
    WellboreDiameter = WellboreDiameter / 12 / 2    #in to ft, then to radius
    P_Wellhead = P_Wellhead                         #psia
    T_Surface     = T_Surface                       #°F
    T_Bottomhole  = T_Bottomhole                    #°F
    t_Production = t_Production * 24.0              #Day(s) to hours
    #Fluid Data
    SG_Oil = SG_Oil                                                 #Dimensionless
    SG_Gas = SG_Gas                                                 #Dimensionless
    GOR = GOR                                                       #SCF/STB
    rho_AnnFluid = rho_AnnFluid                                     #lb/ft³
    rho_ProdFluid = Function__rho_ProdFluid (GOR, SG_Oil, SG_Gas)   #lb/ft³
    mu_AnnFluid = mu_AnnFluid * 0.000671968994813 * 3600            #cP to lb/ft-hr
    mu_ProdFluid = mu_ProdFluid * 0.000671968994813 * 3600          #cP to lb/ft-hr
    mu_Gas = mu_Gas * 0.000671968994813 * 3600                      #cP to lb/ft-hr
    beta_Fluid = 0.000278                                           #Usually on the order of 0.001 to 0.0001 in 1/Kelvin for many liquids; Converted to 1/°F
    #Thermal Conductivities [All in BTU/(hr·ft·°F)]
    k_Earth = k_Earth / 24
    k_Cement = k_Cement / 24
    k_Casing = k_Casing / 24
    k_Tubing = k_Tubing / 24
    k_AnnFluid = k_AnnFluid / 24
    k_ProdFluid = k_ProdFluid / 24
    #Heat Capacities [All in BTU/(lbm·°F)]
    Cp_AnnFluid = 0.94
    Cp_ProdFluid = 0.51
    Cp_Earth = 0.30
    #Earth or Surrounding Formation Density (lbm/ft³)
    rho_Earth = 156.0  

    #For Forced Convection From the Fluid Flow Throughout the Tubing
    #Calculating Prandtl number for the produced fluid
    N_Pr_ProdFluid = Function__N_Pr(Cp_ProdFluid, mu_ProdFluid, k_ProdFluid)
    #Calculating Reynolds number for the produced fluid
    N_Re_ProdFluid = Function__N_Re(Function__rho_ProdFluid (GOR, SG_Oil, SG_Gas), q_ProdFluid, TubingID, mu_ProdFluid)
    #Calculating Nusselt number for a forced convection
    N_Nu_ProdFluid = Function__N_Nu(N_Re_ProdFluid, N_Pr_ProdFluid)
    #Calculating the forced-convection heat transfer coefficient
    h_ProdFluid = Function__h_f(N_Nu_ProdFluid, k_ProdFluid, TubingID)

    #For Natural Convection in the Annulus Where Stagnant Fluid is Present
    #Calculating Prandtl number for the annulus fluid
    N_Pr_AnnFluid = Function__N_Pr(Cp_AnnFluid, mu_AnnFluid, k_AnnFluid)
    #Calculating Grashof number for the annulus fluid
    Guess_DeltaT = 10.0
    N_Gr_AnnFluid = Function__N_Gr(CasingID, TubingOD, rho_AnnFluid, beta_Fluid, mu_AnnFluid, Guess_DeltaT)
    #Calculating annular convective heat transfer coefficient
    h_AnnFluid = Function__h_ann(N_Gr_AnnFluid, N_Pr_AnnFluid, k_AnnFluid, TubingOD, CasingID)

    #For Thermal Resistance by Conduction in the Surrounding Earth
    #Calculating t_Dw parameter
    t_Dw = Function__t_Dw(k_Earth, t_Production, rho_Earth, Cp_Earth, WellboreDiameter)
    #Calculating f(t) parameter
    f_t = Function__f_t(t_Dw)

    #The Overall Heat Transfer Coefficient
    U = Function__U(TubingOD, TubingID, h_ProdFluid, k_Tubing, h_AnnFluid, CasingID, k_Casing, CasingOD, k_Cement, WellboreDiameter, k_Earth, f_t)

    #Iteration Criteria
    Tolerance = 1e-2
    MaxIterations = 1000
    dL = 100  #The control volume is set at every 100 ft from the bottomhole to the surface

    #Initializing lists to store the values
    Depths = []
    Final_Tf_next = []
    Final_U = []
    Final_q = []

    #Starting point from the bottomhole and keeping to move up to the surface
    CurrentDepth = TVD
    T_f_i = T_Bottomhole
    GeothermalGradient = 0.02  # °F/ft
    
    while CurrentDepth >= 0:
        for i in range(MaxIterations):
            #Recomputing N_Gr with the updated guess
            N_Gr_AnnFluid = Function__N_Gr(CasingID, TubingOD, rho_AnnFluid, beta_Fluid, mu_AnnFluid, Guess_DeltaT)
            h_AnnFluid = Function__h_ann(N_Gr_AnnFluid, N_Pr_AnnFluid, k_AnnFluid, TubingOD, CasingID)

            #Recomputing the overall heat transfer coefficient
            U = Function__U(TubingOD, TubingID, h_ProdFluid, k_Tubing, h_AnnFluid, CasingID, k_Casing, CasingOD, k_Cement, WellboreDiameter, k_Earth, f_t)

            #Calculating the overall heat transfer rate
            q = Function__q(TubingOD, CurrentDepth, U, T_f_i, T_Surface, CurrentDepth, GeothermalGradient)

            #Calculating the mass flow rate
            m_dot = Function__m_dot(q_ProdFluid, rho_AnnFluid)

            #Calculating the heat transfer rate through the annulus fluid
            Q_ann = Function__Q_ann(CasingID, CurrentDepth, h_AnnFluid, Guess_DeltaT)

            #For the Target of Tf_i+1
            T_f_avg = Function__T_f_average(T_f_i, Q_ann, m_dot, Cp_ProdFluid, Function__Fc(P_Wellhead, m_dot, GOR, SG_Oil, SG_Gas, T_f_i, T_Surface, CurrentDepth), Function__Potential(Cp_ProdFluid), dL)
            T_f_next = (2 * T_f_avg) - T_f_i

            #The Recalculated Delta Temperature
            Recalc_DeltaT = (m_dot * Cp_ProdFluid * (T_f_i - T_f_next)) / (2 * np.pi * CurrentDepth * CasingID * h_AnnFluid)

            if abs(Recalc_DeltaT - Guess_DeltaT) < Tolerance:
                break

            Guess_DeltaT = Recalc_DeltaT

        #Storing the results
        Depths.append(CurrentDepth)
        Final_Tf_next.append(T_f_next)
        Final_U.append(U)
        Final_q.append(q)

        #Update for the next interval
        T_f_i = T_f_next
        CurrentDepth -= dL

    #Showing the final results to the user
    ActualGeothermalGradient = (T_Bottomhole - T_Surface) / TVD
    FormationTemperature = [T_Bottomhole - (ActualGeothermalGradient * (TVD - depth)) for depth in Depths]
    #Table
    TableContent = {"Depth (ft)": Depths,
                    "Formation Temperature (°F)": FormationTemperature,
                    "Fluid Temperature (°F)": Final_Tf_next,
                    "Overall Heat Transfer Coeff. (BTU/ft²-hr-°F)": Final_U,
                    "Overall Heat Transfer Rate (BTU/hr)": Final_q,}
    TableContent_DataFrame = pd.DataFrame(TableContent)
    TableContent_DataFrame.index = range(1, len(TableContent_DataFrame) + 1)
    StyledTable = (TableContent_DataFrame.style
                   .format({"Depth (ft)": "{:.0f}",
                            "Formation Temperature (°F)": "{:.2f}",
                            "Fluid Temperature (°F)": "{:.2f}",
                            "Overall Heat Transfer Coeff. (BTU/ft²-hr-°F)": "{:.6f}",
                            "Overall Heat Transfer Rate (BTU/hr)": "{:.3f}", })
                   .set_properties(**{'text-align': 'center'})
                   .set_table_styles([{'selector': 'th',
                                       'props': [('font-weight', 'bold'),('background-color', '#f0f0f0')]},
                                      {'selector': 'td', 'props': [('text-align', 'center')]}]))
    st.header("Results: Heat Transfer Calculation Recap")
    st.dataframe(StyledTable, use_container_width = True)
    
    #Chart
    ChartContent = {"Depth (ft)": Depths,
                    "Fluid Temperature (°F)": Final_Tf_next,
                    "Formation Temperature (°F)": FormationTemperature,}
    ChartContent_DataFrame = pd.DataFrame(ChartContent)
    ChartModel = px.line(ChartContent_DataFrame, x = ["Fluid Temperature (°F)", "Formation Temperature (°F)"],
                         y = "Depth (ft)", title = "Temperature Profile Towards Depth",
                         labels = {"value": "Temperature (°F)", "variable": "Legend"}, hover_name = "Depth (ft)")    
    ChartModel.update_layout(yaxis_title = "Depth (ft)", xaxis_title = "Temperature (°F)", xaxis = dict(side = "top"),
                             yaxis = dict(autorange = "reversed"), legend_title = "Legend", hovermode = "x unified",
                             margin = dict(t = 150))
    st.plotly_chart(ChartModel, use_container_width = True)



#STREAMLIT END USER INTERFACE - MAIN CONTENT

#Selecting between single phase and multiphase fluid flow cases
FlowType = st.sidebar.radio("Select your fluid flow type here:", ("Single Phase Incompressible", "Multiphase"))
#Setting up the content header
st.header(f"{WellType} - {FlowType}")

#Setting up the main interface between production and drilling cases
if WellType == "Drilling Well Case":
     st.markdown("<h1 style='text-align: center; color: red;'>This feature is coming soon.</h1>", unsafe_allow_html=True)
else:
    #Setting up the main interface between single phase and multiphase fluid
    if FlowType == "Multiphase":
        #Presenting the typical well schematic illustration
        ImagePath = "WellSchematic.jpg"
        ViewImage = Image.open(ImagePath)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(ViewImage, caption='Your typical well schematic (not to scale).', width = 200)
        #Setting up the users' input section
        #Presenting default values at the user interface (Values are originally in the Oil Field Untis)
        if "WellData" not in st.session_state:
            st.session_state.WellData = {
                "TVD": 5355.0,
                "q_ProdFluid": 500,
                "SG_Oil": 0.85,
                "SG_Gas": 1.04,
                "rho_AnnFluid": 63.96,
                "mu_AnnFluid": 1.025,
                "mu_ProdFluid": 3, #Assuming light oil
                "mu_Gas": 0.01,
                "GOR": 500, #Assuming light oil
                "TubingID": 2.875,
                "TubingOD": 3.0,
                "CasingID": 6.46,
                "CasingOD": 7.0,
                "WellboreDiameter": 9.0,
                "P_Wellhead": 113.0,
                "T_Surface": 76.0,
                "T_Bottomhole": 108.0,
                "t_Production": 6.58,
            }
        if "ThermalData" not in st.session_state:
            st.session_state.ThermalData = {
                "k_Earth": 33.3,
                "k_Cement": 20.4,
                "k_Casing": 500.0,
                "k_Tubing": 500.0,
                "k_AnnFluid": 13.88,
                "k_ProdFluid": 2.16,
                "Cp_AnnFluid": 0.94,
                "Cp_ProdFluid": 0.51,
                "rho_Earth": 156.0,
                "Cp_Earth": 0.30,
            }
        #Well Data Inputs
        st.subheader("General Well Data Input")
        st.write(f"Current Unit System: **Oil Field Units**")
        col1, col2 = st.columns(2)
        with col1:
            q_ProdFluid = st.number_input(f"Well Production Rate ({Units["q_ProdFluid"]})", value=st.session_state.WellData["q_ProdFluid"])
            P_Wellhead = st.number_input(f"Wellhead Pressure ({Units["P_Wellhead"]})", value=st.session_state.WellData["P_Wellhead"])
            t_Production = st.number_input(f"Production Time ({Units["t_Production"]})", value=st.session_state.WellData["t_Production"])
            TubingID = st.number_input(f"Tubing Inner Diameter ({Units["TubingID"]})", value=st.session_state.WellData["TubingID"])
            TubingOD = st.number_input(f"Tubing Outer Diameter ({Units["TubingOD"]})", value=st.session_state.WellData["TubingOD"])
            T_Surface = st.number_input(f"Surface Temperature ({Units["T_Surface"]})", value=st.session_state.WellData["T_Surface"])             
        with col2:
            CasingID = st.number_input(f"Production Casing Inner Diameter ({Units["CasingID"]})", value=st.session_state.WellData["CasingID"])
            CasingOD = st.number_input(f"Production Casing Outer Diameter ({Units["CasingOD"]})", value=st.session_state.WellData["CasingOD"])
            WellboreDiameter = st.number_input(f"Production Hole Size ({Units["WellboreDiameter"]})", value=st.session_state.WellData["WellboreDiameter"])
            TVD = st.number_input(f"True Vertical Depth ({Units["TVD"]})", value=st.session_state.WellData["TVD"])
            T_Bottomhole = st.number_input(f"Bottomhole Temperature ({Units["T_Bottomhole"]})", value=st.session_state.WellData["T_Bottomhole"])
        #Fluid Data Inputs
        st.subheader("Fluid Data Input")
        st.write(f"Current Unit System: **Oil Field Units**")
        col3, col4 = st.columns(2)
        with col3:
            SG_Oil = st.number_input(f"Oil Specific Gravity ({Units["SG_Oil"]})", value=st.session_state.WellData["SG_Oil"])
            SG_Gas = st.number_input(f"Gas Specific Gravity ({Units["SG_Gas"]})", value=st.session_state.WellData["SG_Gas"])
            mu_ProdFluid = st.number_input(f"Produced Fluid Viscosity ({Units["mu_ProdFluid"]})", value=st.session_state.WellData["mu_ProdFluid"])          
            mu_Gas = st.number_input(f"Gas Viscosity ({Units["mu_Gas"]})", value=st.session_state.WellData["mu_Gas"])
        with col4:
            mu_AnnFluid = st.number_input(f"Annulus or Completion Fluid Viscosity ({Units["mu_AnnFluid"]})", value=st.session_state.WellData["mu_AnnFluid"])
            rho_AnnFluid = st.number_input(f"Annulus or Completion Fluid Density ({Units["rho_AnnFluid"]})", value=st.session_state.WellData["rho_AnnFluid"])
            GOR = st.number_input(f"Gas-Oil Ratio ({Units["GOR"]})", value=st.session_state.WellData["GOR"])
        #Thermal Data Inputs
        st.subheader("Thermal Data Input")
        st.write(f"Current Unit System: **Oil Field Units**")
        col5, col6 = st.columns(2)
        with col5:
            k_AnnFluid = st.number_input(f"Annulus or Completion Fluid Thermal Conductivity ({Units["k_AnnFluid"]})", value=st.session_state.ThermalData["k_AnnFluid"])
            k_ProdFluid = st.number_input(f"Produced Fluid Thermal Conductivity ({Units["k_ProdFluid"]})", value=st.session_state.ThermalData["k_ProdFluid"])
            k_Tubing = st.number_input(f"Tubing Thermal Conductivity ({Units["k_Tubing"]})", value=st.session_state.ThermalData["k_Tubing"])
            k_Casing = st.number_input(f"Casing Thermal Conductivity ({Units["k_Casing"]})", value=st.session_state.ThermalData["k_Casing"])
            k_Cement = st.number_input(f"Cement Thermal Conductivity ({Units["k_Cement"]})", value=st.session_state.ThermalData["k_Cement"])
            k_Earth = st.number_input(f"Earth Thermal Conductivity ({Units["k_Earth"]})", value=st.session_state.ThermalData["k_Earth"])
        with col6:
            Cp_AnnFluid = st.number_input(f"Annulus or Completion Fluid Specific Heat ({Units["Cp_AnnFluid"]})", value=st.session_state.ThermalData["Cp_AnnFluid"])
            Cp_ProdFluid = st.number_input(f"Produced Fluid Specific Heat ({Units["Cp_ProdFluid"]})", value=st.session_state.ThermalData["Cp_ProdFluid"])
            Cp_Earth = st.number_input(f"Earth or Surrounding Formation Specific Heat ({Units["Cp_Earth"]})", value=st.session_state.ThermalData["Cp_Earth"])
            rho_Earth = st.number_input(f"Earth or Surrounding Formation Density ({Units["rho_Earth"]})", value=st.session_state.ThermalData["rho_Earth"])

        #Updating the default values into users' determined values as the new input
        st.session_state.WellData = {
            "TVD": TVD,
            "q_ProdFluid": q_ProdFluid,
            "SG_Oil": SG_Oil,
            "SG_Gas": SG_Gas,
            "rho_AnnFluid": rho_AnnFluid,
            "mu_AnnFluid": mu_AnnFluid,
            "mu_ProdFluid": mu_ProdFluid,
            "mu_Gas": mu_Gas,
            "GOR": GOR,
            "t_Production": t_Production,
            "TubingID": TubingID,
            "TubingOD": TubingOD,
            "CasingID": CasingID,
            "CasingOD": CasingOD,
            "WellboreDiameter": WellboreDiameter,
            "P_Wellhead": P_Wellhead,
            "T_Surface": T_Surface,
            "T_Bottomhole": T_Bottomhole,
        }

        st.session_state.ThermalData = {
            "k_Earth": k_Earth,
            "k_Cement": k_Cement,
            "k_Casing": k_Casing,
            "k_Tubing": k_Tubing,
            "k_AnnFluid": k_AnnFluid,
            "k_ProdFluid": k_ProdFluid,
            "Cp_AnnFluid": Cp_AnnFluid,
            "Cp_ProdFluid": Cp_ProdFluid,
            "rho_Earth": rho_Earth,
            "Cp_Earth": Cp_Earth,
        }

        #Running the heat transfer simulation
        main_ProductionMultiphase(
            TVD=st.session_state.WellData["TVD"],
            q_ProdFluid=st.session_state.WellData["q_ProdFluid"],
            CasingID=st.session_state.WellData["CasingID"],
            CasingOD=st.session_state.WellData["CasingOD"],
            TubingID=st.session_state.WellData["TubingID"],
            TubingOD=st.session_state.WellData["TubingOD"],
            WellboreDiameter=st.session_state.WellData["WellboreDiameter"],
            P_Wellhead=st.session_state.WellData["P_Wellhead"],
            T_Surface=st.session_state.WellData["T_Surface"],
            T_Bottomhole=st.session_state.WellData["T_Bottomhole"],
            t_Production=st.session_state.WellData["t_Production"],
            SG_Oil=st.session_state.WellData["SG_Oil"],
            SG_Gas=st.session_state.WellData["SG_Gas"],
            rho_AnnFluid=st.session_state.WellData["rho_AnnFluid"],
            mu_AnnFluid=st.session_state.WellData["mu_AnnFluid"],
            mu_ProdFluid=st.session_state.WellData["mu_ProdFluid"],
            mu_Gas=st.session_state.WellData["mu_Gas"],
            GOR=st.session_state.WellData["GOR"],
            k_Earth=st.session_state.ThermalData["k_Earth"],
            k_Cement=st.session_state.ThermalData["k_Cement"],
            k_Casing=st.session_state.ThermalData["k_Casing"],
            k_Tubing=st.session_state.ThermalData["k_Tubing"],
            k_AnnFluid=st.session_state.ThermalData["k_AnnFluid"],
            k_ProdFluid=st.session_state.ThermalData["k_ProdFluid"],
            Cp_AnnFluid=st.session_state.ThermalData["Cp_AnnFluid"],
            Cp_ProdFluid=st.session_state.ThermalData["Cp_ProdFluid"],
            rho_Earth=st.session_state.ThermalData["rho_Earth"],
            Cp_Earth=st.session_state.ThermalData["Cp_Earth"]
        )
        
    else:
        st.markdown("<h1 style='text-align: center; color: red;'>This feature is coming soon.</h1>", unsafe_allow_html=True)
