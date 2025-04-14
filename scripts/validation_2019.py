# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 14:35:00 2025

@author: aless
"""

import pypsa 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd
import cartopy.crs as ccrs



# Capacità installata
def capacity_installed(n, nation, capacity):
    
    generators = n.generators[n.generators.bus.str.startswith(nation)]
    
    pv = generators.loc[n.generators.carrier == 'solar'].p_nom_opt.sum() /1e3
    onwind = generators.loc[n.generators.carrier == 'onwind'].p_nom_opt.sum() /1e3
    offwind = generators.loc[n.generators.carrier.isin(['offwind-ac', 'offwind-dc', 'offwind-float']), 'p_nom_opt'].sum() / 1e3
    
    capacity['PyPSA'] = [pv,onwind,offwind]
    
    xlabel = capacity.index
    ylabel = 'Capacity [GW]'
    title = 'Comparison of renewable installed capacity in 2040'
    
    plot_histogram(capacity, n, xlabel, ylabel, title, nation)
    
    return capacity


    

# Produzione

def production(n, nation, prod):    

    generators_t = n.generators_t.p.loc[:, n.generators_t.p.columns.str.startswith(nation)]
    storage_units_t = n.storage_units_t.p_dispatch.loc[:, n.storage_units_t.p_dispatch.columns.str.startswith(nation)]    
    
    
    #
    coal = generators_t.filter(like = None, regex="coal|lignite").sum().sum() / 1e6
    oil = generators_t.filter(like = "oil").sum().sum() / 1e6
    natural_gas = generators_t.filter(like = "CCGT").sum().sum() / 1e6
    nuclear = generators_t.filter(like = "nuclear").sum().sum() / 1e6
    hydro = storage_units_t.sum().sum() / 1e6 + generators_t.filter(like = "ror").sum().sum() / 1e6
    biomass = generators_t.filter(like = "biomass").sum().sum() / 1e6
    wind = generators_t.filter(like=None, regex='wind').sum().sum()/1e6
    solar = generators_t.filter(like = "solar").sum().sum() / 1e6
    geothermal = generators_t.filter(like = "geothermal").sum().sum() / 1e6
    
    pypsa_prod = {
        'Coal': coal,
        'Oil': oil,
        'Natural gas': natural_gas,
        'Nuclear': nuclear,
        'Hydro': hydro,
        'Biofuels': biomass,
        'Wind': wind,
        'Solar PV': solar,
        'Geothermal': geothermal
    }
    
    prod['PyPSA'] = prod.index.map(pypsa_prod).fillna(0)
    
    xlabel = prod.index
    ylabel = 'Total generation [TWh]'
    title = 'Comparison of total generation per carrier in 2019'
    
    plot_histogram(prod, n, xlabel, ylabel, title, nation)
    
    return prod


def total_load(n):
    population = {
        'AT': 8879920,
        'SI': 2120460,
        'IT': 59729090,
        'CH': 8575280,
        'FR': 67382060,
        'ME': 622030,
        'GR': 10721580
        }
    
    load = {
        'AT': 8.342,
        'SI': 7.143,
        'IT': 5.26,
        'CH': 7.354,
        'FR': 7.011,
        'ME': 5.133,
        'GR': 5.118
        }
    
    iea = {k: population[k] * load[k] for k in population} 
    load = pd.DataFrame.from_dict(iea, orient='index', columns=['IEA']) /1e6
    
    for nation in ['FR', 'CH', 'AT', 'GR', 'SI', 'ME', 'IT']:
        loads_t = n.loads_t.p.loc[:, n.loads_t.p.columns.str.startswith(nation)]
        load.at[nation, 'PyPSA'] = loads_t.sum().sum() / 1e6
    
    xlabel = load.index
    ylabel = 'Total load [TWh]'
    title = 'Validation of total load 2019'
    nation = 'comparison_load'
    
    plot_histogram(load, n, xlabel, ylabel, title, nation)
    
    return load

def prod_italy(n):
    generators_t = n.generators_t.p.loc[:, n.generators_t.p.columns.str.startswith('IT')]
    storage_units_t = n.storage_units_t.p_dispatch.loc[:, n.storage_units_t.p_dispatch.columns.str.startswith('IT')]
    
    it = generators_t.sum().sum() / 1e6 + storage_units_t.sum().sum() / 1e6
    
    return it

    

def prod_load(prod, load):
    prod = prod.sort_index()
    load = load.sort_index()
    
    prod['Load'] = load['PyPSA']
    
    xlabel = load.index
    ylabel = 'Energy [TWh]'
    title = 'Comparison between total production and load in 2019'
    nation = 'production_load'
    
    plot_histogram(prod, n, xlabel, ylabel, title, nation)
    
    
def compute_expected_emissions_2040(n):
    """
    Calcola le emissioni attese al 2040 a partire dai dati PyPSA del 2019.

    Args:
        n (pypsa.Network): il network PyPSA con risultati del 2019.

    Returns:
        dict: contiene le emissioni 2019, stimate 1990 e target 2040.
    """
    # Emissioni 2019 (tonnellate orarie per generatore)
    emissions = (
        n.generators_t.p
        .div(n.generators.efficiency, axis=1)
        .mul(n.generators.carrier.map(n.carriers.co2_emissions), axis=1)
    )

    total_emissions_2019 = emissions.sum().sum()  # tonnellate totali nel 2019

    # Se 2019 è il 75.53% del 1990:
    total_emissions_1990 = total_emissions_2019 / 0.7553

    # Target 2040 = 22.35% del 1990
    target_emissions_2040 = total_emissions_1990 * 0.2235

    return {
        "emissions_2019_tonnes": total_emissions_2019 /1e6,
        "estimated_emissions_1990_tonnes": total_emissions_1990 /1e6,
        "target_emissions_2040_tonnes": target_emissions_2040 /1e6
    }

    

def plot_useful_things(n):
    # 1. Trova le linee tra IT e SI
    interconnection_mask = (
        (n.lines.bus0.str.startswith("IT") & n.lines.bus1.str.startswith("SI")) |
        (n.lines.bus1.str.startswith("IT") & n.lines.bus0.str.startswith("SI"))
    )
    interconnection_lines = n.lines[interconnection_mask].index
    
    # 2. Calcola il flusso netto da IT a SI (o viceversa)
    power_exchange = n.lines_t.p0[interconnection_lines].sum(axis=1)
    
    # 3. Trova generatori in SI
    si_generators = n.generators[
        n.generators.bus.str.startswith("SI")
    ]
    
    # Filtra CCGT e coal
    ccgt_gen = si_generators[si_generators.carrier.str.contains("CCGT", case=False)]
    coal_gen = si_generators[si_generators.carrier.str.contains("coal|lignite", case=False)]
    
    # 4. Estrai la produzione oraria
    ccgt_profile = n.generators_t.p[ccgt_gen.index].sum(axis=1)
    coal_profile = n.generators_t.p[coal_gen.index].sum(axis=1)
    
    # 5. Plot
    plt.figure(figsize=(15, 5))
    plt.plot(power_exchange, label="Power exchange IT <-> SI [MW]", color="red")
    plt.plot(ccgt_profile, label="CCGT production in SI [MW]", color="blue")
    plt.plot(coal_profile, label="Coal production in SI [MW]", color="black")
    plt.xlabel("Time")
    plt.ylabel("Power [MW]")
    plt.title("Power exchange IT-SI and Slovenian generation (CCGT & Coal)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    output_folder = "C:\\Users\\aless\\Desktop\\PhD_Pisa\\2025_01_04\\validation_europe\\results"
    plt.savefig(f"{output_folder}/Slovenia_fluxes.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_import_export_SIAT(n):
    # 1. Linee tra AT e SI
    at_si_mask = (
        (n.lines.bus0.str.startswith("AT") & n.lines.bus1.str.startswith("SI")) |
        (n.lines.bus1.str.startswith("AT") & n.lines.bus0.str.startswith("SI"))
    )
    at_si_lines = n.lines[at_si_mask].index
    
    # 2. Flusso orario totale sulle linee (positivo: dal bus0 al bus1)
    power_exchange_at_si = n.lines_t.p0[at_si_lines].sum(axis=1)
    
    # 3. Generatori per Austria e Slovenia
    at_generators = n.generators[n.generators.bus.str.startswith("AT")]
    si_generators = n.generators[n.generators.bus.str.startswith("SI")]
    
    # 4. Filtra per CCGT e coal
    ccgt_at = at_generators[at_generators.carrier.str.contains("ccgt", case=False)]
    ccgt_si = si_generators[si_generators.carrier.str.contains("ccgt", case=False)]
    coal_at = at_generators[at_generators.carrier.str.contains("coal", case=False)]
    coal_si = si_generators[si_generators.carrier.str.contains("coal", case=False)]
    
    # 5. Profili di generazione
    ccgt_at_profile = n.generators_t.p[ccgt_at.index].sum(axis=1)
    ccgt_si_profile = n.generators_t.p[ccgt_si.index].sum(axis=1)
    coal_at_profile = n.generators_t.p[coal_at.index].sum(axis=1)
    coal_si_profile = n.generators_t.p[coal_si.index].sum(axis=1)
    
    # 6. Plot
    plt.figure(figsize=(16, 6))
    plt.plot(power_exchange_at_si, label="Power exchange AT <-> SI [MW]", color="orange", linewidth=2)
    plt.plot(ccgt_at_profile, label="CCGT Austria [MW]", linestyle="--", color="blue")
    plt.plot(ccgt_si_profile, label="CCGT Slovenia [MW]", linestyle="--", color="skyblue")
    plt.plot(coal_at_profile, label="Coal Austria [MW]", linestyle="--", color="black")
    plt.plot(coal_si_profile, label="Coal Slovenia [MW]", linestyle="--", color="grey")
    
    plt.xlabel("Time")
    plt.ylabel("Power [MW]")
    plt.title("Power exchange AT-SI and generation profiles (CCGT & Coal)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    output_folder = "C:\\Users\\aless\\Desktop\\PhD_Pisa\\2025_01_04\\validation_europe\\results"
    plt.savefig(f"{output_folder}/AT_SI_fluxes.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_SI(n):
    # 1. Filtra i load della Slovenia
    si_loads = n.loads[n.loads.bus.str.startswith("SI")].index
    
    # 2. Estrai il profilo orario
    si_load_profile = n.loads_t.p_set[si_loads].sum(axis=1)
    
    # 3. Plot
    plt.figure(figsize=(14, 5))
    plt.plot(si_load_profile, label="Slovenia Load [MW]", color="green")
    plt.xlabel("Time")
    plt.ylabel("Load [MW]")
    plt.title("Hourly Electricity Load - Slovenia")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    output_folder = "C:\\Users\\aless\\Desktop\\PhD_Pisa\\2025_01_04\\validation_europe\\results"
    plt.savefig(f"{output_folder}/Load_SI.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
        

def plot_histogram(df, n, xlabel, ylabel, title, nation):
    #grafico confronto capacità
    x_axis = df.index

    # Posizioni per le barre
    x = np.arange(len(x_axis))

    # Larghezza delle barre
    bar_width = 0.35  

    # Crea il grafico
    fig, ax = plt.subplots(figsize=(10, 6))

    # Barre
    bars1 = ax.bar(x - bar_width / 2, df[df.columns[0]].values.flatten(), width=bar_width, label=df.columns[0], color='#1f77b4', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + bar_width / 2, df[df.columns[1]].values.flatten(), width=bar_width, label=df.columns[1], color='#ff7f0e', edgecolor='black', linewidth=1)

    # Etichette e titolo

    ax.set_ylabel(ylabel, fontsize = 14, fontweight = "bold")
    ax.set_title(title, fontsize=14, fontweight='bold')


    # Assegna le etichette formattate all'asse X
    ax.set_xticks(x)
    ax.set_xticklabels(xlabel, rotation=30, ha="right", fontsize=10, fontstyle='italic', color='#2F4F4F')  # Rotazione a 30°

    # Aggiungere la griglia
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Aggiungere i valori sopra le barre
    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom', fontsize=10, color='black')

    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom', fontsize=10, color='black')

    # Legenda
    ax.legend(fontsize=10, frameon=True, framealpha=0.7, facecolor='white', edgecolor='black')

    # Layout
    plt.tight_layout(pad=3.0)

    # Mostra il grafico
    output_folder = "C:\\Users\\aless\\Desktop\\PhD_Pisa\\2025_01_04\\validation_europe\\results"
    plt.savefig(f"{output_folder}/{nation}.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


# Main
if __name__ == "__main__":
    n = pypsa.Network("C:\\Users\\aless\\Desktop\\PhD_Pisa\\2025_01_03\\pypsa-europe\\validation50nodes\\2019_validation.nc")
    
    load = total_load(n)
    
    diff_load = (load['IEA'] - load['PyPSA']).sum()
    
    diff_production = 0
    total_production = {}
    
    for nation in ['FR', 'CH', 'AT', 'GR', 'SI', 'ME']:
        prod = pd.read_excel("C:\\Users\\aless\\Desktop\\PhD_Pisa\\2025_01_04\\validation_europe\\countries.xlsx", sheet_name=nation, index_col=0)
        prod = prod.loc[prod.Year == 2019, ['Value']] / 1e3
        prod = prod.rename(columns={'Value': 'IEA'})  # rinomina la colonna
        prod = prod.drop(['Other sources', 'Waste', 'Tide'], errors='ignore')
       
        prod = production(n, nation, prod)
        diff_production += (prod['IEA'] - prod['PyPSA']).sum()
        
        total_production[nation] = prod['PyPSA'].sum()
        
    total_production = pd.DataFrame.from_dict(total_production, orient='index', columns=['Production'])
    total_production.loc['IT'] = prod_italy(n)
    
    prod_load(total_production, load)
    emissions = compute_expected_emissions_2040(n)
    # plot_useful_things(n)
    # plot_import_export_SIAT(n)
    # plot_SI(n)