# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 15:07:15 2022

@author: Ishtiak
"""

# description	: The freight rail decarbonization cost model is used to estimate freight rail levelized costs and carbon intensity
#                 for alternative powertrain decarbonization technologies
# version		: 1.0.0
# python_version: 3.8
# status		: in progress
#updates tried in this version from v6: Completely change the visualization page. Added tooltip info for the visualization elements. It is not complete yet. Also, the options have been capitalized 
#and unwanted symbols like _ have been removed.Still needs to add the carbon intensity vis plots
#requirements
import time
from functools import reduce
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image
#import matplotlib.pyplot as plt

version = 1
#set page config
st.set_page_config(page_title=None, page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)

#define tab names and numbers
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["About", "Input Files", "Input Data Display", "Run Model","Visualize Levelized Cost",
                                        "Visualize Carbon Intensity"])

# %% dataframe conversion function
#this function converts the output data to utf-8 encoded data. It is important to download the outputs
@st.cache
def convert_df(df):
    
   return df.to_csv().encode('utf-8')


# %% visualization function
#Only needed if you want to show a graph of the input file
def interactive_plot(df):
    
    x_axis_val = st.selectbox('Select the X-axis', options=df.columns)
    y_axis_val = st.selectbox('Select the Y-axis', options=df.columns)

    plot = px.line(df, x=x_axis_val, y=y_axis_val)
    st.plotly_chart(plot, use_container_width=True)

#this function takes the region/yearly/techno cost, yaxis value, and the desired
# y axis title for the output visualization page.
#horizontal bar plot    
def bar_ploth(df,yaxis,ylab): 
    #the output is in wide data format. Tidy it to long
    df_long = pd.melt(df, id_vars=yaxis, value_vars=["levelized_fuel_cost (cents/ton-mile)", 
                                                                    'levelized_infrastructure_cost (cents/ton-mile)', 
       'levelized_contingency_cost (cents/ton-mile)','levelized_train_cost (cents/ton-mile)'])
    fig = px.bar(df_long, y=yaxis, 
                     x='value',
                     color = 'variable',
                     labels = {'value' : "Cost (cents/ton-mile)", yaxis : ylab},
                     title=f"Levelized cost variation by {ylab}")
    #to change the legend key, first create a dictionary of the existing and the desired names
    newnames = {'levelized_fuel_cost (cents/ton-mile)':'Fuel', 'levelized_infrastructure_cost (cents/ton-mile)': 'Infrastructure',
                'levelized_contingency_cost (cents/ton-mile)' : "Contingency", 'levelized_train_cost (cents/ton-mile)': 'Train'}
    #the following
    fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                      legendgroup = newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])
                                     )
                  )
    
    st.plotly_chart(fig, use_container_width=True)
    
#vertical bar plot     
def bar_plotv(df,yaxis,ylab): 
    #the output is in wide data format. Tidy it to long
    df_long = pd.melt(df, id_vars=yaxis, value_vars=["levelized_fuel_cost (cents/ton-mile)", 
                                                                    'levelized_infrastructure_cost (cents/ton-mile)', 
       'levelized_contingency_cost (cents/ton-mile)','levelized_train_cost (cents/ton-mile)'])
    fig = px.bar(df_long, x=yaxis, 
                     y='value',
                     color = 'variable',
                     labels = {'value' : "Cost (cents/ton-mile)", yaxis : ylab},
                     title=f"Levelized cost variation by {ylab}")
    #to change the legend key, first create a dictionary of the existing and the desired names
    newnames = {'levelized_fuel_cost (cents/ton-mile)':'Fuel', 'levelized_infrastructure_cost (cents/ton-mile)': 'Infrastructure',
                'levelized_contingency_cost (cents/ton-mile)' : "Contingency", 'levelized_train_cost (cents/ton-mile)': 'Train'}
    #the following
    fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                      legendgroup = newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])
                                     )
                  )
    
    st.plotly_chart(fig, use_container_width=True)
    
#this function takes the region/yearly/techno carbon intensity, yaxis value, and the desired
# y axis title for the output visualization page.
#horizontal bar plot    
def C_bar_ploth(df,yaxis,ylab): 
    #the output is in wide data format. Tidy it to long
    df_long = pd.melt(df, id_vars=yaxis, value_vars=["CO2_carbon_intensity (kgCO2/ton-mile)", 
                                                                    'CH4_carbon_intensity (kgCH4/ton-mile)'])
    fig = px.bar(df_long, y=yaxis, 
                     x='value',
                     color = 'variable',
                     labels = {'value' : "Carbon Intensity (kg CO2 eq/ton-mile)", yaxis : ylab},
                     title=f"Carbon intensity variation by {ylab}")
    #to change the legend key, first create a dictionary of the existing and the desired names
    newnames = {'CO2_carbon_intensity (kgCO2/ton-mile)':'CO2', 'CH4_carbon_intensity (kgCH4/ton-mile)': 'CH4'}
    #the following
    fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                      legendgroup = newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])
                                     )
                  )
    
    st.plotly_chart(fig, use_container_width=True)
    
#vertical bar plot     
def C_bar_plotv(df,yaxis,ylab): 
    #the output is in wide data format. Tidy it to long
    df_long = pd.melt(df, id_vars=yaxis, value_vars=["CO2_carbon_intensity (kgCO2/ton-mile)", 
                                                                    'CH4_carbon_intensity (kgCH4/ton-mile)'])
    fig = px.bar(df_long, x=yaxis, 
                     y='value',
                     color = 'variable',
                     labels = {'value' : "Carbon Intensity (kg CO2 eq/ton-mile)", yaxis : ylab},
                     title=f"Carbon intensity variation by {ylab}")
    #to change the legend key, first create a dictionary of the existing and the desired names
    newnames = {'CO2_carbon_intensity (kgCO2/ton-mile)':'CO2', 'CH4_carbon_intensity (kgCH4/ton-mile)': 'CH4'}
    #the following
    fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                      legendgroup = newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])
                                     )
                  )
    
    st.plotly_chart(fig, use_container_width=True)
    
# %% Model function
def model(df1,df2,df3,df4,df5,df_parameter):
    dfs = [df1, df2, df3, df4, df5]
    df_final = reduce(lambda left, right: pd.merge(left, right, on=['technology', 'region','energy_system_scenario','freight_demand_scenario','year']), dfs)
    
    i = np_parameter[0, 1] # annual discount rate (percentage point)
    J = np_parameter[1, 1] # technology lifetime (year)
    c = np_parameter[2, 1] # contingency factor (%)
    
    for n in df_final.index:
        LN = df_final.loc[n, 'number_of_locomotives'] # network-level number of locomotive
        PN = df_final.loc[n, 'number_of_powertrains'] # network-level number of powertrains
        TN = df_final.loc[n, 'number_of_tender_cars'] # network-level number of tender cars
        FN = df_final.loc[n, 'number_of_freight_cars'] # network-level number of freight cars
        LC = df_final.loc[n, 'cost_per_locomotive ($)'] # locomotive unit cost ($/unit)
        PC = df_final.loc[n, 'cost_per_powertrain ($)'] # powertrain unit cost ($/unit)
        TC = df_final.loc[n, 'cost_per_tender_car ($)']  # tender car unit cost ($/unit)
        FC = df_final.loc[n, 'cost_per_freight_car ($)']  # freight car unit cost ($/unit)
        OMT = df_final.loc[n, 'train_O&M_cost ($/year)']  # train operation & maintenance cost ($/year)
        FP = df_final.loc[n, 'energy_price'] # fuel or energy price ($/gallon, $/kg-H2, or $/kWh-e)
        FU = df_final.loc[n, 'energy_consumption'] # fuel or energy usage (gallon, kg-H2, or kWh-e)
        RN = df_final.loc[n, 'number_of_charging_stations'] # network-level number of refueling or recharging stations
        RSC = df_final.loc[n, 'cost_per_station ($)'] # refueling or recharging station unit cost ($/unit)
        TRM = df_final.loc[n, 'track_mile (mi)'] # freight rail track miles
        RTC = df_final.loc[n, 'rail_track_cost ($/mi)'] # rail track cost ($/mile)
        OMI = df_final.loc[n, 'infrastructure_O&M_cost ($/year)'] # infrastructure operation & maintenance cost ($/year)
        TMT = df_final.loc[n, 'freight_ton_mile_travel'] # freignt ton-mile travel (ton-mile/year)
        LTC = (LN * LC + PN * PC + TN * TC + FN * FC + OMT * J)/(1 + i)**J/((TMT * J)/(1 + i)**J) * 100 # estimate levelized train cost (¢/ton-mile)
        df_final.loc[n, 'levelized_train_cost (cents/ton-mile)'] = LTC
        LFC = FP * FU * J/(1 + i)**J/((TMT * J)/(1 + i)**J) * 100 # estimate levelized fuel cost (¢/ton-mile)
        df_final.loc[n, 'levelized_fuel_cost (cents/ton-mile)'] = LFC
        LIC = (RN * RSC + TRM * RTC + OMI * J)/(1 + i)**J/((TMT * J)/(1 + i)**J) * 100 # estimate levelized infrastructure cost (¢/ton-mile)
        df_final.loc[n, 'levelized_infrastructure_cost (cents/ton-mile)'] = LIC
        LCC = (LTC + LFC + LIC) * c/100 # estimate levelized contingency cost (¢/ton-mile)
        df_final.loc[n, 'levelized_contingency_cost (cents/ton-mile)'] = LCC
        TLC = LTC + LFC + LIC + LCC # estimate total levelized cost (¢/ton-mile)
        df_final.loc[n, 'total_levelized_cost (cents/ton-mile)'] = TLC
        CO2U = df_final.loc[n, 'upstream_CO2_emission_factor'] # upstream CO2 emission factor (kgCO2/gallon, kgCO2/kg-H2, or kgCO2/kWh-e)
        CO2D = df_final.loc[n, 'downstream_CO2_emission_factor'] # downstream CO2 emission factor (kgCO2/gallon, kgCO2/kg-H2, or kgCO2/kWh-e)
        CH4U = df_final.loc[n, 'upstream_CH4_emission_factor'] # upstream CH4 emission factor (kgCH4/gallon, kgCH4/kg-H2, or kgCH4/kWh-e)
        CH4D = df_final.loc[n, 'downstream_CH4_emission_factor']  # downstream CH4 emission factor (kgCH4/gallon, kgCH4/kg-H2, or kgCH4/kWh-e)
        LCO2 = (CO2U + CO2D) * FU * J / (1 + i) ** J / ((TMT * J) / (1 + i) ** J)  # estimate levelized CO2 carbon intensity (kgCO2/ton-mile)
        df_final.loc[n, 'CO2_carbon_intensity (kgCO2/ton-mile)'] = LCO2
        LCH4 = (CH4U + CH4D) * FU * J / (1 + i) ** J / ((TMT * J) / (1 + i) ** J)  # estimate levelized CH4 carbon intensity (kgCH4/ton-mile)
        df_final.loc[n, 'CH4_carbon_intensity (kgCH4/ton-mile)'] = LCH4
        if df_final.loc[n, 'technology'] == 'biodiesel':
            LCO2eq = LCO2 + 27.0 * LCH4 # estimate levelized CO2-equivalent carbon intensity based on GWP-100 for CH4-non fossil from IPCC AR6
        else:
            LCO2eq = LCO2 + 29.8 * LCH4 # estimate levelized CO2-equivalent carbon intensity based on GWP-100 for CH4-fossil from IPCC AR6
        df_final.loc[n, 'CO2eq_carbon_intensity (kgCO2eq/ton-mile)'] = LCO2eq
    
    levelized_cost = df_final.drop(df_final.columns[[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,35,36,37]],axis=1)
    carbon_intensity = df_final.drop(df_final.columns[5:35],axis=1) # drop unnecessary columns for the carbon intensity output file
    return levelized_cost, carbon_intensity

# %% This section is to avoid name error until either all necessary data are uploaded or pre-stored data are chosen
try: df1
except NameError: df1 = None
try: df2
except NameError: df2 = None
try: df3
except NameError: df3 = None
try: df4
except NameError: df4 = None
try: df5
except NameError: df5 = None
try: df_parameter
except NameError: df_parameter = None

# %% Read hard-coded files
#working directory. Change if necessary
#direct = "G:\\.shortcut-targets-by-id\\1m_vWVSXCB3I9mQi1vuOqJsIjp9HUbG1y\\Rail Decarbonization\\Task 5 Cost Analysis\\Python code\\cost_model_v1\\"

train_pre = pd.read_csv("a_data_file_train_consist.csv")
energ_pre = pd.read_csv("b_data_file_energy_consumption.csv")
inf_pre = pd.read_csv("c_data_file_infrastructure.csv")
price_pre = pd.read_csv("d_data_file_energy_price.csv")
dem_pre = pd.read_csv("e_data_file_freight_demand.csv")
param_pre = pd.read_csv("f_data_file_parameter.csv")

#Read hard-coded pdf files
with open("Cost Model Documentation Report.pdf", "rb") as pdf_file:
    PDFbyte = pdf_file.read()

#add sample output image. This image is hardcoded. Please change the directory based on where this image files is stored in your pc.
image = Image.open("CM Sample Output.png")



# %% About page
page_title = "A-STEP: Achieving Sustainable Train Energy Pathways"
body = "A-STEP: Achieving Sustainable Train Energy Pathways"
subhead = f"Cost Model (v{version})"

with tab1:
    #st.set_page_config(page_title=page_title)
    st.header(body)
    st.subheader(subhead)

    #add Description pdf        
    st.download_button(
       label = "Click to download the description of the cost model",
       data = PDFbyte,
       file_name = "Cost Model Documentation.pdf",
       mime='application/octet-stream',
       key='Description'
    )

    #add sample output image
    st.image(image, caption='Sample output: Technology vs Cost')

# %%input page
with tab2:
    st.title("")
    st.subheader("Either upload a file or use pre-stored data for each input. Choose from the drop-down menus")
    
    col1, col2, col3 = st.columns(3)

    #create a dropdown meny to choose upload file or use pre-stored data. One for each of the six inputs, i.e., option1 through option6
    
    #Dropdown options for train consist data
    with col1:
        st.header("Train Consist Input")
        option1 = st.selectbox(
             label = '', options = ('Use pre-stored data (click to download)','Upload formatted data'),key = 1)
        if option1 == 'Upload formatted data':
            traindf = st.file_uploader("Upload your input Train Consist file", type=["csv"])
            if traindf is not None:
                df1 = pd.read_csv(traindf)
        elif option1 == 'Use pre-stored data (click to download)':
            st.download_button(
                "Download pre-stored train consist data",
                convert_df(train_pre),
                "train consist.csv",
                "text/csv",
                key='ps1'
            )
            df1 = train_pre
        
    #Dropdown options for energy consumtion data
        st.header("Energy Consumption Input")
        option2 = st.selectbox(
             '',
             ('Use pre-stored data (click to download)','Upload formatted data'),key = 2)
        if option2 == 'Upload formatted data':
            energ = st.file_uploader("Upload your input Energy Consumption file", type=["csv"])
            if energ is not None:
                df2 = pd.read_csv(energ)
        elif option2 == 'Use pre-stored data (click to download)':
            st.download_button(
                "Download pre-stored Energy Consumption data",
                convert_df(energ_pre),
                "evenrgy consume.csv",
                "text/csv",
                key='ps2'
            )
            df2 = energ_pre
    with col2:   
        #Dropdown options for infrastructure data
        st.header("Infrastructure Input")
        option3 = st.selectbox(
             '',
             ('Use pre-stored data (click to download)','Upload formatted data'),key = 3)
        if option3 == 'Upload formatted data':
            inf = st.file_uploader("Upload your input Infrastructure file", type=["csv"])
            if inf is not None:
                df3 = pd.read_csv(energ)
        elif option3 == 'Use pre-stored data (click to download)':
            st.download_button(
                "Download pre-stored Infrastructure data",
                convert_df(inf_pre),
                "Infrastructure.csv",
                "text/csv",
                key='ps3'
            )
            df3 = inf_pre
        
        #Dropdown options for energy price data
        st.header("Energy Price Input")
        option4 = st.selectbox(
             '',
             ('Use pre-stored data (click to download)','Upload formatted data'),key = 4)
        if option4 == 'Upload formatted data':
            price = st.file_uploader("Upload your input Energy Price file", type=["csv"])
            if price is not None:
                df4 = pd.read_csv(price)
        elif option4 == 'Use pre-stored data (click to download)':
            st.download_button(
                "Download pre-stored Energy Price data",
                convert_df(price_pre),
                "Energy price.csv",
                "text/csv",
                key='ps4'
            )
            df4 = price_pre
    with col3:    
        #Dropdown options for Freight Demand data
        st.header("Freight Demand Input")
        option5 = st.selectbox(
             '',
             ('Use pre-stored data (click to download)','Upload formatted data'),key = 5)
        if option5 == 'Upload formatted data':
            dem = st.file_uploader("Upload your input Freight Demand file", type=["csv"])
            if dem is not None:
                df5 = pd.read_csv(dem)
        elif option5 == 'Use pre-stored data (click to download)':
            st.download_button(
                "Download pre-stored Freight Demand data",
                convert_df(dem_pre),
                "Freight demand.csv",
                "text/csv",
                key='ps5'
            )
            df5 = dem_pre
        
        #Dropdown options for Parameters data
        st.header("Parameter Value Input")
        option6 = st.selectbox(
             '',
             ('Use pre-stored data (click to download)','Upload formatted data'),key = 6)
        if option6 == 'Upload formatted data':
            param = st.file_uploader("Upload your input Parameters file", type=["csv"])
            if param is not None:
                df_parameter = pd.read_csv(param)
        elif option6 == 'Use pre-stored data (click to download)':
            st.download_button(
                "Download pre-stored Parameters data",
                convert_df(param_pre),
                "Parameters.csv",
                "text/csv",
                key='ps6'
            )
            df_parameter = param_pre
    
# %% design  display page      

with tab3:
    st.title("")
    
    
    if df1  is None or df2 is None or df3 is None or df4 is None or df5 is None or df_parameter is None:
        st.warning("Upload all necessary data first. Go back to the Input Page")
    else:   
        # st.header('Show input train consist data')
        # interactive_plot(df1)
        #
        st.header('Show Energy Consumption Data')
        st.write(df2)
            
        st.header('Show Infrastructure Data')
        st.write(df3)
            
        st.header('Show Evergy Price Data')
        st.write(df4)
    
        st.header('Uploaded Frieght Demand Data')
        st.write(df5)
        
        np_parameter = np.array(df_parameter)
        st.header('Uploaded Parameter Data')
        st.write(df_parameter)
    

# %% design model page
with tab4:
    st.title("Run Simulation")
    
    @st.experimental_memo()
    def computation():
        start = time.time()
        levelized_cost, carbon_intensity = model(df1,df2,df3,df4,df5,df_parameter)
        end = time.time()
        levelized_cost_df = convert_df(levelized_cost)
        carbon_intensity_df = convert_df(carbon_intensity)
        return levelized_cost_df, carbon_intensity_df,start,end,levelized_cost,carbon_intensity
    if 'run' not in st.session_state:
         st.session_state.run = 0
    #st.info(f"state = {st.session_state.run}")     

    if df1  is None or df2 is None or df3 is None or df4 is None or df5 is None or df_parameter is None:
        st.warning("Upload all necessary data first. Go back to the Input Page")
    else:
        run = st.button('Press to run the simulation')
        if run:
            st.session_state.run = 1
        if st.session_state.run == 1:  
            levelized_cost_df, carbon_intensity_df,start,end,levelized_cost,carbon_intensity = computation()
            st.info(
               f"Run time = {round(end-start,ndigits = 2)} seconds"
            )
            st.download_button(
                "Download Levelized Cost Data",
                levelized_cost_df,
                f"levelized cost_{version}.csv",
                "text/csv",
                key='download-csv'
            )
            st.download_button(
                "Download Carbon Intensity Data",
                carbon_intensity_df,
                f"carbon intensity_{version}.csv",
                "text/csv",
                key='download-profile'
            )
# %% Output visualization page
with tab5:
    st.title("Visualize levelized cost outputs")    
    col1, col2 = st.columns(2)
    if st.session_state.run == 0:
        st.warning("Run the Simulation First to Display Outputs") 
    elif st.session_state.run == 1:
        with col1:
            st.subheader("Plot levelized cost outputs")  
            option_costplot = st.selectbox(label = "Choose a graph", 
                                           options = ("None", "Cost by Regions","Cost by Technologies", "Cost by Years","Cost by Energy Systems","Cost by Freight Demand"))
            if option_costplot == "Cost by Regions":
                #inputs for the first plot: Region vs cost
                ener_sc1 = st.selectbox('Select energy system scenario', options=levelized_cost.energy_system_scenario.unique(),
                                       help = "Dummy tooltip",key = 11, format_func = lambda x: (x.replace('_', ' ')).title())
                freight_sc1 = st.selectbox('Select freight demand scenario', options=levelized_cost.freight_demand_scenario.unique(),
                                       help = "Dummy tooltip",key = 12, format_func = lambda x: (x.replace('_', ' ')).title())
                tech_sc1 = st.selectbox('Select power train technology', options=levelized_cost.technology.unique(),
                                       help = "Dummy tooltip",key = 13, format_func = lambda x: (x.replace('_', ' ')).title())
                year_sc1 = st.selectbox('Select year', options=levelized_cost.year.unique(),
                                       help = "Dummy tooltip",key = 14)
                #calculations for creating the plot
        
                region_cost = levelized_cost[
                                      (levelized_cost.energy_system_scenario == ener_sc1)&  # select energy system scenario
                                      (levelized_cost.freight_demand_scenario == freight_sc1)&  # select freight demand scenario
                                      (levelized_cost.technology == tech_sc1)&  # select technology
                                      # (df_cost.region == 'north_east')&  # select region
                                      (levelized_cost.year == year_sc1)  # select year
                                     ]
                bar_ploth(region_cost, "region","Regions")

            elif option_costplot == "Cost by Technologies":
                #inputs for the second plot: cost vs technology
                st.subheader("Inputs for displaying levelized cost variation by technology")       
    
                ener_sc2 = st.selectbox('Select energy system scenario', options=levelized_cost.energy_system_scenario.unique(),
                                       help = "Dummy tooltip",key = 21, format_func = lambda x: (x.replace('_', ' ')).title())
                freight_sc2 = st.selectbox('Select freight demand scenario', options=levelized_cost.freight_demand_scenario.unique(),
                                       help = "Dummy tooltip",key = 22, format_func = lambda x: (x.replace('_', ' ')).title())
                reg_sc2 = st.selectbox('Select power train technology', options=levelized_cost.region.unique(),
                                       help = "Dummy tooltip",key = 23, format_func = lambda x: (x.replace('_', ' ')).title())
                year_sc2 = st.selectbox('Select year', options=levelized_cost.year.unique(),
                                       help = "Dummy tooltip",key = 24)
                
                tech_cost = levelized_cost[
                                    (levelized_cost.energy_system_scenario == ener_sc2)&  # select energy system scenario
                                    (levelized_cost.freight_demand_scenario == freight_sc2)&  # select freight demand scenario
                                    # (df_cost.technology == 'diesel')&  # select technology
                                    (levelized_cost.region == reg_sc2)&  # select region
                                    (levelized_cost.year == year_sc2)  # select year
                                    ]
                bar_plotv(tech_cost,'technology',"Technology")
            
            elif option_costplot == "Cost by Years":
                #inputs for the third plot: cost vs Years
                st.subheader("Inputs for displaying levelized cost variation by years")       
    
                ener_sc3 = st.selectbox('Select energy system scenario', options=levelized_cost.energy_system_scenario.unique(),
                                       help = "Business As Usual means no new federal policies \n\n50% reduction by 2050 means___\n\nNet zero by 2050 means__",
                                       key = 31, format_func = lambda x: (x.replace('_', ' ')).title())
                #the last part, i.e., format function is to remove underscores and capitalize for the display options
                freight_sc3 = st.selectbox('Select freight demand scenario', options=levelized_cost.freight_demand_scenario.unique(),
                                       help = "Dummy tooltip",key = 32, format_func = lambda x: (x.replace('_', ' ')).title())
                reg_sc3 = st.selectbox('Select power train technology', options=levelized_cost.region.unique(),
                                       help = "Dummy tooltip",key = 33, format_func = lambda x: (x.replace('_', ' ')).title())
                tech_sc3 = st.selectbox('Select power train technology', options=levelized_cost.technology.unique(),
                                      help = "Dummy tooltip",key = 34, format_func = lambda x: (x.replace('_', ' ')).title())
                
                year_cost = levelized_cost[
                                    (levelized_cost.energy_system_scenario == ener_sc3)&  # select energy system scenario
                                    (levelized_cost.freight_demand_scenario == freight_sc3)&  # select freight demand scenario
                                    (levelized_cost.technology == tech_sc3)&  # select technology
                                    (levelized_cost.region == reg_sc3)  # select region
                                    # (df_cost.year == 2025)  # select year
                                    ]
                year_cost['year'] = year_cost['year'].astype('category') #for numerical y axis, convert to factor
                bar_plotv(year_cost,'year',"Year")

            elif option_costplot == "Cost by Energy Systems":
                #the last part, i.e., format function is to remove underscores and capitalize for the display options
                freight_sc4 = st.selectbox('Select freight demand scenario', options=levelized_cost.freight_demand_scenario.unique(),
                                       help = "Dummy tooltip",key = 41, format_func = lambda x: (x.replace('_', ' ')).title())
                reg_sc4 = st.selectbox('Select power train technology', options=levelized_cost.region.unique(),
                                       help = "Dummy tooltip",key = 42, format_func = lambda x: (x.replace('_', ' ')).title())
                tech_sc4 = st.selectbox('Select power train technology', options=levelized_cost.technology.unique(),
                                      help = "Dummy tooltip",key = 43, format_func = lambda x: (x.replace('_', ' ')).title())
                year_sc4 = st.selectbox('Select year', options=levelized_cost.year.unique(),
                                       help = "Dummy tooltip",key = 44)
                
                ESS_cost = levelized_cost[
                                    (levelized_cost.year == year_sc4)&  # select energy system scenario
                                    (levelized_cost.freight_demand_scenario == freight_sc4)&  # select freight demand scenario
                                    (levelized_cost.technology == tech_sc4)&  # select technology
                                    (levelized_cost.region == reg_sc4)  # select region
                                    ]
                bar_plotv(ESS_cost,'energy_system_scenario',"Energy System Scenario")
            elif option_costplot == "Cost by Freight Demand":
                #the last part, i.e., format function is to remove underscores and capitalize for the display options
                ener_sc5 = st.selectbox('Select energy system scenario', options=levelized_cost.energy_system_scenario.unique(),
                                       help = "Business As Usual means no new federal policies \n\n50% reduction by 2050 means___\n\nNet zero by 2050 means__",
                                       key = 51, format_func = lambda x: (x.replace('_', ' ')).title())
                reg_sc5 = st.selectbox('Select power train technology', options=levelized_cost.region.unique(),
                                       help = "Dummy tooltip",key = 52, format_func = lambda x: (x.replace('_', ' ')).title())
                tech_sc5 = st.selectbox('Select power train technology', options=levelized_cost.technology.unique(),
                                      help = "Dummy tooltip",key = 53, format_func = lambda x: (x.replace('_', ' ')).title())
                year_sc5 = st.selectbox('Select year', options=levelized_cost.year.unique(),
                                       help = "Dummy tooltip",key = 54)
                FDS_cost = levelized_cost[
                                    (levelized_cost.energy_system_scenario == ener_sc5)&  # select energy system scenario
                                    # (df_cost.freight_demand_scenario == 'business_as_usual')&  # select freight demand scenario
                                    (levelized_cost.technology == tech_sc5)&  # select technology
                                    (levelized_cost.region == reg_sc5)&  # select region
                                    (levelized_cost.year == year_sc5)  # select year
                                  ]
                
                bar_plotv(FDS_cost,'freight_demand_scenario',"Freight demand scenario")


        
# %% Output visualization page
with tab6:
    st.title("Visualize carbon intensity outputs")
    #calculate equivalent c02 from the output file
    
    col1, col2 = st.columns(2)
    if st.session_state.run == 0:
        st.warning("Run the Simulation First to Display Outputs") 
    elif st.session_state.run == 1:
        for n in carbon_intensity.index:
            LCH4 = carbon_intensity.loc[n, 'CH4_carbon_intensity (kgCH4/ton-mile)']  # CH4 carbon intensity (kgCH4/ton-mile)
            if carbon_intensity.loc[n, 'technology'] == 'biodiesel':
                LCH4_CO2eq = 27.0 * LCH4 # CH4 carbon intensity in terms of CO2eq (kgCO2eq/ton-mile) for non-fossil sources
            else:
                LCH4_CO2eq = 29.8 * LCH4 # CH4 carbon intensity in terms of CO2eq (kgCO2eq/ton-mile) for fossil sources
            carbon_intensity.loc[n, 'CH4_carbon_intensity (kgCH4/ton-mile)'] = LCH4_CO2eq

        with col1:
            st.subheader("Plot carbon intensity outputs")  
            option_CIplot = st.selectbox(label = "Choose a graph", 
                                           options = ("None", "Carbon intensity by Regions","Carbon intensity by Technologies", 
                                                      "Carbon intensity by Years","Carbon intensity by Energy Systems",
                                                      "Carbon intensity by Freight Demand"))
            if option_CIplot == "Carbon intensity by Regions":
                #inputs for the first plot: Region vs cost
                co2_ener_sc1 = st.selectbox('Select energy system scenario', options=carbon_intensity.energy_system_scenario.unique(),
                                       help = "Dummy tooltip",key = "C11", format_func = lambda x: (x.replace('_', ' ')).title())
                co2_freight_sc1 = st.selectbox('Select freight demand scenario', options = carbon_intensity.freight_demand_scenario.unique(),
                                       help = "Dummy tooltip",key = "C12", format_func = lambda x: (x.replace('_', ' ')).title())
                co2_tech_sc1 = st.selectbox('Select power train technology', options = carbon_intensity.technology.unique(),
                                       help = "Dummy tooltip",key = "C13", format_func = lambda x: (x.replace('_', ' ')).title())
                co2_year_sc1 = st.selectbox('Select year', options = carbon_intensity.year.unique(),
                                       help = "Dummy tooltip",key = "C14")
                #calculations for creating the plot
                region_CI = carbon_intensity[
                                  (carbon_intensity.energy_system_scenario == co2_ener_sc1)&  # select an energy system scenario
                                  (carbon_intensity.freight_demand_scenario == co2_freight_sc1)&  # select a freight demand scenario
                                  (carbon_intensity.technology == co2_tech_sc1)&  # select a technology
                                  # (df_CI == 'north_east')&  # select a region
                                  (carbon_intensity.year == co2_year_sc1)  # select a year
                                  ]
                
                C_bar_ploth(region_CI, "region","Regions")

            elif option_CIplot == "Carbon intensity by Technologies":
                #inputs for the second plot: cost vs technology
    
                co2_ener_sc2 = st.selectbox('Select energy system scenario', options=carbon_intensity.energy_system_scenario.unique(),
                                       help = "Dummy tooltip",key = "C21", format_func = lambda x: (x.replace('_', ' ')).title())
                co2_freight_sc2 = st.selectbox('Select freight demand scenario', options=carbon_intensity.freight_demand_scenario.unique(),
                                       help = "Dummy tooltip",key = "C22", format_func = lambda x: (x.replace('_', ' ')).title())
                co2_reg_sc2 = st.selectbox('Select power train technology', options=carbon_intensity.region.unique(),
                                       help = "Dummy tooltip",key = "C23", format_func = lambda x: (x.replace('_', ' ')).title())
                co2_year_sc2 = st.selectbox('Select year', options=carbon_intensity.year.unique(),
                                       help = "Dummy tooltip",key = "C24")
                
                tech_CI = carbon_intensity[
                                    (carbon_intensity.energy_system_scenario == co2_ener_sc2)&  # select energy system scenario
                                    (carbon_intensity.freight_demand_scenario == co2_freight_sc2)&  # select freight demand scenario
                                    # (df_cost.technology == 'diesel')&  # select technology
                                    (carbon_intensity.region == co2_reg_sc2)&  # select region
                                    (carbon_intensity.year == co2_year_sc2)  # select year
                                    ]
                C_bar_plotv(tech_CI,'technology',"Technology")
            
            elif option_CIplot == "Carbon intensity by Years":
                #inputs for the third plot: cost vs Years
    
                co2_ener_sc3 = st.selectbox('Select energy system scenario', options=carbon_intensity.energy_system_scenario.unique(),
                                       help = "Business As Usual means no new federal policies \n\n50% reduction by 2050 means___\n\nNet zero by 2050 means__",
                                       key = "C31", format_func = lambda x: (x.replace('_', ' ')).title())
                #the last part, i.e., format function is to remove underscores and capitalize for the display options
                co2_freight_sc3 = st.selectbox('Select freight demand scenario', options=carbon_intensity.freight_demand_scenario.unique(),
                                       help = "Dummy tooltip",key = "C32", format_func = lambda x: (x.replace('_', ' ')).title())
                co2_reg_sc3 = st.selectbox('Select power train technology', options=carbon_intensity.region.unique(),
                                       help = "Dummy tooltip",key = "C33", format_func = lambda x: (x.replace('_', ' ')).title())
                co2_tech_sc3 = st.selectbox('Select power train technology', options=carbon_intensity.technology.unique(),
                                      help = "Dummy tooltip",key = "C34", format_func = lambda x: (x.replace('_', ' ')).title())
                
                year_CI = carbon_intensity[
                                    (carbon_intensity.energy_system_scenario == co2_ener_sc3)&  # select energy system scenario
                                    (carbon_intensity.freight_demand_scenario == co2_freight_sc3)&  # select freight demand scenario
                                    (carbon_intensity.technology == co2_tech_sc3)&  # select technology
                                    (carbon_intensity.region == co2_reg_sc3)  # select region
                                    # (df_cost.year == 2025)  # select year
                                    ]
                year_CI['year'] = year_CI['year'].astype('category') #for numerical y axis, convert to factor
                C_bar_plotv(year_CI,'year',"Year")

            elif option_CIplot == "Carbon intensity by Energy Systems":
                #inputs for the third plot: cost vs Years
    
                
                #the last part, i.e., format function is to remove underscores and capitalize for the display options
                co2_freight_sc4 = st.selectbox('Select freight demand scenario', options=carbon_intensity.freight_demand_scenario.unique(),
                                       help = "Dummy tooltip",key = "C41", format_func = lambda x: (x.replace('_', ' ')).title())
                co2_reg_sc4 = st.selectbox('Select power train technology', options=carbon_intensity.region.unique(),
                                       help = "Dummy tooltip",key = "C42", format_func = lambda x: (x.replace('_', ' ')).title())
                co2_tech_sc4 = st.selectbox('Select power train technology', options=carbon_intensity.technology.unique(),
                                      help = "Dummy tooltip",key = "C43", format_func = lambda x: (x.replace('_', ' ')).title())
                co2_year_sc4 = st.selectbox('Select year', options=carbon_intensity.year.unique(),
                                       help = "Dummy tooltip",key = "C44")
                
                ESS_CI = carbon_intensity[
                                    (carbon_intensity.year == co2_year_sc4)&  # select energy system scenario
                                    (carbon_intensity.freight_demand_scenario == co2_freight_sc4)&  # select freight demand scenario
                                    (carbon_intensity.technology == co2_tech_sc4)&  # select technology
                                    (carbon_intensity.region == co2_reg_sc4)  # select region
                                    ]
                C_bar_plotv(ESS_CI,'energy_system_scenario',"Energy System Scenario")
                
            elif option_CIplot == "Carbon intensity by Freight Demand":
                #inputs for the third plot: cost vs Years
                
                co2_ener_sc5 = st.selectbox('Select energy system scenario', options=carbon_intensity.energy_system_scenario.unique(),
                                       help = "Business As Usual means no new federal policies \n\n50% reduction by 2050 means___\n\nNet zero by 2050 means__",
                                       key = "C54", format_func = lambda x: (x.replace('_', ' ')).title())
                co2_reg_sc5 = st.selectbox('Select power train technology', options=carbon_intensity.region.unique(),
                                       help = "Dummy tooltip",key = "C51", format_func = lambda x: (x.replace('_', ' ')).title())
                co2_tech_sc5 = st.selectbox('Select power train technology', options=carbon_intensity.technology.unique(),
                                      help = "Dummy tooltip",key = "C52", format_func = lambda x: (x.replace('_', ' ')).title())
                co2_year_sc5 = st.selectbox('Select year', options=carbon_intensity.year.unique(),
                                       help = "Dummy tooltip",key = "C53")
                
                
                FDS_CI = carbon_intensity[
                                    (carbon_intensity.year == co2_year_sc5)&  # select energy system scenario
                                    (carbon_intensity.energy_system_scenario == co2_ener_sc5)&  # select freight demand scenario
                                    (carbon_intensity.technology == co2_tech_sc5)&  # select technology
                                    (carbon_intensity.region == co2_reg_sc5)  # select region
                                    ]
                C_bar_plotv(FDS_CI,'freight_demand_scenario',"Freight Demand Scenario")
                                  


        


    
    
    
    
    

    
