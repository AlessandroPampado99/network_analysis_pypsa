# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 10:15:21 2025

@author: aless
"""

import pypsa 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd
import cartopy.crs as ccrs



# Capacità FER installata
def fer_capacity_installed(n):
    fer_capacity = pd.DataFrame([121.0, 34.0, 15.1], index=['solar', 'on-wind', 'off-wind'], columns=['DE-IT']) # GW
    
    generators = n.generators[n.generators.bus.str.startswith("IT")]
    
    pv = generators.loc[n.generators.carrier == 'solar'].p_nom_opt.sum() /1e3
    onwind = generators.loc[n.generators.carrier == 'onwind'].p_nom_opt.sum() /1e3
    offwind = generators.loc[n.generators.carrier.isin(['offwind-ac', 'offwind-dc', 'offwind-float']), 'p_nom_opt'].sum() / 1e3
    
    fer_capacity['PyPSA'] = [pv,onwind,offwind]
    
    xlabel = fer_capacity.index
    ylabel = 'Capacity [GW]'
    title = 'Comparison of renewable installed capacity in 2040'
    
    plot_histogram(fer_capacity, n, xlabel, ylabel, title)
    
    return fer_capacity


# Totale storage

def total_storage(n):
    storage_terna = 166.5 # GWh
    
    storage_units = n.storage_units[n.storage_units.bus.str.startswith("IT")]
    stores = n.stores[n.stores.bus.str.startswith("IT")]
    
    storage_pypsa = (storage_units.p_nom_opt * storage_units.max_hours).sum()/1e3 + stores.e_nom_opt.sum()/1e3
    
    error = (storage_pypsa - storage_terna) / storage_terna * 100
    return error


# % FER su totale

def fer_percentage(n, fer_capacity):
    fer_perc_terna = 84 # %
    
    generators = n.generators[n.generators.bus.str.startswith("IT")]
    storage_units = n.storage_units[n.storage_units.bus.str.startswith("IT")]
    
    fer_capacity = (fer_capacity['PyPSA'].sum() +
                    generators.loc[n.generators.carrier == 'biomass'].p_nom_opt.sum() /1e3 +
                    generators.loc[n.generators.carrier == 'ror'].p_nom_opt.sum() /1e3 +
                    generators.loc[n.generators.carrier == 'geothermal'].p_nom_opt.sum() /1e3 +
                    storage_units.p_nom_opt.sum() / 1e3
                    )
    total_capacity = generators.p_nom_opt.sum() / 1e3 + storage_units.p_nom_opt.sum() / 1e3
    
    fer_perc_pypsa = fer_capacity / total_capacity * 100
    
    error = (fer_perc_pypsa - fer_perc_terna) / fer_perc_terna * 100
    
    return error
    

# Produzione

def production(n):    
    prod = pd.DataFrame([46,168,121,17,59,6], index=['hydro', 'solar', 'wind', 'FER other', 'natural gas', 'non-FER other'], columns=['DE-IT']) # TWh

    generators_t = n.generators_t.p.loc[:, n.generators_t.p.columns.str.startswith('IT')]
    storage_units_t = n.storage_units_t.p_dispatch.loc[:, n.storage_units_t.p_dispatch.columns.str.startswith('IT')]    
    
    hydro = storage_units_t.sum().sum() / 1e6 + generators_t.filter(like = "ror").sum().sum() / 1e6
    solar = generators_t.filter(like = "solar").sum().sum() / 1e6
    wind = generators_t.filter(like=None, regex='wind').sum().sum()/1e6
    biomass = generators_t.filter(like =None, regex="biomass|geothermal").sum().sum() / 1e6
    natural_gas = generators_t.filter(like = "CCGT").sum().sum() / 1e6
    oil = generators_t.filter(like = "oil").sum().sum() / 1e6
    
    prod['PyPSA'] = [hydro, solar, wind, biomass, natural_gas, oil]
    
    xlabel = prod.index
    ylabel = 'Total generation [TWh]'
    title = 'Comparison of total generation per carrier in 2040'
    
    plot_histogram(prod, n, xlabel, ylabel, title)
    
    return prod


# Import export

def import_nations(n):
    import_export = pd.DataFrame([6,27,31,-3,-3,-3], index=['AT', 'CH', 'FR', 'GR', 'ME', 'SI'], columns=['DE-IT'])
    
    import_lines = dict()
    for nation in ['FR', 'CH', 'AT', 'SI', 'GR', 'ME']:
        mask = (
        (n.lines.bus0.str.startswith("IT") & n.lines.bus1.str.startswith(nation)) |
        (n.lines.bus0.str.startswith(nation) & n.lines.bus1.str.startswith("IT"))
        )
        import_lines[nation] = n.lines[mask]
        import_lines[nation] = n.lines_t.p0[import_lines[nation].index].sum().sum()/1e6
        
    for nation in ['FR', 'CH', 'AT', 'SI', 'GR', 'ME']:
        mask = (
        (n.links.bus0.str.startswith("IT") & n.links.bus1.str.startswith(nation)) |
        (n.links.bus0.str.startswith(nation) & n.links.bus1.str.startswith("IT"))
        )
        link = n.links[mask]
        import_lines[nation] += n.links_t.p0[link.index].sum().sum()/1e6
        


    import_lines = pd.DataFrame.from_dict(import_lines, orient='index')
    import_export['PyPSA'] = import_lines
    
    xlabel = import_export.index
    ylabel = 'Imported energy [TWh]'
    title = 'Comparison of import/export flows in 2040'
    
    plot_histogram(import_export, n, xlabel, ylabel, title)
    
    return import_export


def plot_histogram(df, n, xlabel, ylabel, title):
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
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 2.5, round(yval, 2), ha='center', va='bottom', fontsize=10, color='black')

    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 2.5, round(yval, 2), ha='center', va='bottom', fontsize=10, color='black')

    # Legenda
    ax.legend(fontsize=10, frameon=True, framealpha=0.7, facecolor='white', edgecolor='black')

    # Layout
    plt.tight_layout(pad=3.0)

    # Mostra il grafico
    plt.show()


# Main
if __name__ == "__main__":
    n = pypsa.Network("../networks/2040_DEIT.nc")
    fer_capacity = fer_capacity_installed(n)
    error_storage = total_storage(n)
    error_fer = fer_percentage(n, fer_capacity)
    prod = production(n)
    import_export = import_nations(n)

