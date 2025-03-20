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
    "P_Bottomhole": "psi",
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

#Annular convective heat transfer coefficient (hr-ft²-°F)
def Function__h_ann (delta_T):
    delta_T = max(delta_T, 0.1)
    N_Gr = 71200000 * ((CasingID - TubingOD)**3) * (rho_AnnFluid**2) * beta_Fluid * delta_T / (mu_AnnFluid**2)
    h_ann = 0.049 * (N_Gr * N_Pr)**0.33 * (N_Pr**0.074) * k_AnnFluid / (TubingOD * math.log(CasingID / TubingOD))
    return h_ann

#Convection thermal resistance (hr-ft²-°F/BTU)
def Function__R_conv (h_ann):
    return 1 / (2 * math.pi * CasingID * dL * h_ann)

#Total thermal resistance (hr-ft²-°F/BTU)
def Function__R_total(h_ann):
    return Function__R_conv(h_ann) + R_additional

#Fluid density calculator (lb/ft³)
def Function__rho_ProdFluid (GOR, SG_Oil, SG_Gas, P_Bottomhole, T_Bottomhole):
    WaterDensity = 62.4         #lb/ft³ at standard conditions
    AirDensity = 0.0765         #lb/ft³ at standard conditions
    OilBarrelVolume = 5.615     #STB to standard ft³; Gas volume is assumed as "GOR" SCF in every STB of oil
    R = 10.73                   #psi·ft³/lb-mol·°R (gas constant)
    MolarMassGas = 28.97        #lb/lb-mol (molar mass of gas)
    OilMass = (SG_Oil * WaterDensity) * OilBarrelVolume     #Oil mass in lb unit
    GasMass = (SG_Gas * AirDensity) * GOR                   #Gas mass in lb unit
    T_Res_Rankine = T_Bottomhole + 459.67                   #Conversion into Rankine
    GasMole = GasMass / MolarMassGas
    GasVolume = (GasMole * R * T_Res_Rankine) / P_Bottomhole
    ProdFluidDensity = (OilMass + GasMass) / (OilBarrelVolume + GasVolume)
    return ProdFluidDensity


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
                "q_ProdFluid": 10000,
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
                "P_Bottomhole": 3000,
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
            P_Bottomhole = st.number_input(f"Bottomhole Pressure ({Units["P_Bottomhole"]})", value=st.session_state.WellData["P_Bottomhole"])
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
            "P_Bottomhole": P_Bottomhole,
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

        #MAIN DATA
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
        SG_Oil = SG_Oil                                                                             #Dimensionless
        SG_Gas = SG_Gas                                                                             #Dimensionless
        GOR = GOR                                                                                   #SCF/STB
        rho_AnnFluid = rho_AnnFluid                                                                 #lb/ft³
        rho_ProdFluid = Function__rho_ProdFluid (GOR, SG_Oil, SG_Gas, P_Bottomhole, T_Bottomhole)   #lb/ft³
        mu_AnnFluid = mu_AnnFluid * 0.000671968994813 * 3600                                        #cP to lb/ft-hr
        mu_ProdFluid = mu_ProdFluid * 0.000671968994813 * 3600                                      #cP to lb/ft-hr
        mu_Gas = mu_Gas * 0.000671968994813 * 3600                                                  #cP to lb/ft-hr
        beta_Fluid = 0.000278                                                                       #Usually on the order of 0.001 to 0.0001 in 1/Kelvin for many liquids; Converted to 1/°F
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

        #ADDITIONAL DATA
        #Mass Flow Rate (lbm/hr)
        MassFlowRate = q_ProdFluid * rho_ProdFluid
        #Geothermal Gradient (°F/ft)
        GeothermalGradient = (T_Bottomhole - T_Surface) / TVD
        #Control volume length (ft)
        dL = 100

        #Preliminary Calculation
        #Prandtl number (dimensionless)
        N_Pr = 58.1 * Cp_ProdFluid * mu_AnnFluid / k_AnnFluid
        #Additional conduction resistance (hr-ft²-°F/BTU)
        R_additional = math.log(TubingOD / TubingID) / k_Tubing
        
        #Determining the number of calculation steps based on the well TD
        StepCount = int(TVD / dL)
        #Setting up the depth from well TD to wellhead
        Depths = np.linspace(TVD, 0, StepCount + 1)
        #Setting up the fluid temperature array for calculations
        T_Fluid = np.zeros(StepCount + 1)
        #Setting up the fluid temperature calculation starting point
        T_Bottom = T_Bottomhole
        T_Fluid[0] = T_Bottom
        #Determining the formation temperature profile
        T_FormationProfile = T_Surface + GeothermalGradient * Depths

        #Defining the mass flow rate threshold to implement the acceptable scientific boundary
        MassFlowRate_Threshold = 920 #in lbm/hr
        if MassFlowRate < MassFlowRate_Threshold:
            T_Fluid = T_FormationProfile.copy()
            st.warning(f"⚠️ **Warning:** The mass flow rate is extremely low ({MassFlowRate:.3f} lbm/hr). "
                       " Consequently, the fluid spends more time in the wellbore, and more heat is exchanged with the surrounding formation.")
        else:        
            #Main Calculation
            for i in range(StepCount):
                #Defining the depth of each upper section (next interval)
                NextDepth = Depths[i+1]
                #Formation temperature at the end of every next interval
                T_FormationInterval = T_Surface + GeothermalGradient * NextDepth
                #Managing the temperature difference in each step
                deltaT_Guess = T_Fluid[i] - T_FormationInterval
                #Calculating h_ann with the current temperature difference
                h_ann = Function__h_ann(deltaT_Guess)
                #Calculating the total thermal resistance
                R_total = Function__R_total(h_ann)
                #Calculating the heat loss per hour over the interval
                Q = (T_Fluid[i] - T_FormationInterval) / R_total
                #Calculating the temperature drop over the interval
                dT_Fluid = Q / (MassFlowRate * Cp_ProdFluid)
                #Calculating the final fluid temperature
                T_Fluid[i+1] = T_Fluid[i] - dT_Fluid
            #Identifying the fluid temperature at surface
            T_Fluid_Surface = T_Fluid[-1]

        #Table
        TableContent = {"Depth (ft)": Depths,
                        "Formation Temperature (°F)": T_FormationProfile,
                        "Fluid Temperature (°F)": T_Fluid,}
        TableContent_DataFrame = pd.DataFrame(TableContent)
        TableContent_DataFrame.index = range(1, len(TableContent_DataFrame) + 1)
        StyledTable = (TableContent_DataFrame.style
                       .format({"Depth (ft)": "{:.0f}",
                                "Formation Temperature (°F)": "{:.2f}",
                                "Fluid Temperature (°F)": "{:.2f}"})
                       .set_properties(**{'text-align': 'center'})
                       .set_table_styles([{'selector': 'th',
                                           'props': [('font-weight', 'bold'),('background-color', '#f0f0f0')]},
                                          {'selector': 'td', 'props': [('text-align', 'center')]}]))
        st.header("Results: Heat Transfer Calculation Recap")
        st.dataframe(StyledTable, use_container_width = True)

        #Chart
        ChartContent = {"Depth (ft)": Depths,
                        "Fluid Temperature (°F)": T_Fluid,
                        "Formation Temperature (°F)": T_FormationProfile,}
        ChartContent_DataFrame = pd.DataFrame(ChartContent)
        ChartModel = px.line(ChartContent_DataFrame, x = ["Fluid Temperature (°F)", "Formation Temperature (°F)"],
                             y = "Depth (ft)", title = "Temperature Profile Towards Depth",
                             labels = {"value": "Temperature (°F)", "variable": "Legend"}, hover_name = "Depth (ft)",
                             color_discrete_map={"Fluid Temperature (°F)": "red", "Formation Temperature (°F)": "blue"})    
        ChartModel.update_layout(yaxis_title = "Depth (ft)", xaxis_title = "Temperature (°F)", xaxis_range = [0, max(max(T_Fluid), max(T_FormationProfile))],
                                 xaxis = dict(side = "top"), xaxis_showline = False, yaxis = dict(autorange = "reversed"), yaxis_showline = False,
                                 legend_title = "Legend", hovermode = "x unified", margin = dict(t = 150))
        st.plotly_chart(ChartModel, use_container_width = True)

        #Extra Details
        st.markdown(f"**Calculation Remarks**")
        st.markdown(f"According to your gas-oil ratio (GOR), oil specific gravity, gas specific gravity, and bottomhole pressure and temperature, your **production fluid density** is approximately **{rho_ProdFluid:.3f} lb/ft³**.")
        st.markdown(f"According to your production rate and the processed production fluid density, your **mass flow rate** used in this calculation is approximately **{MassFlowRate:.3f} lbm/hr**.")
            
    else:
        st.markdown("<h1 style='text-align: center; color: red;'>This feature is coming soon.</h1>", unsafe_allow_html=True)

