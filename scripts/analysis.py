import pypsa
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pypsa.plot import add_legend_patches
import cartopy.crs as ccrs
import cartopy
import os
import sys
from pathlib import Path
import numpy as np

# Determina il percorso assoluto della directory corrente
current_dir = os.path.dirname(os.path.abspath(__file__))

# Aggiungi il percorso relativo alla directory `../config`
config_path = os.path.abspath(os.path.join(current_dir, "../config"))

# Aggiungi il percorso a sys.path
sys.path.append(config_path)


from config import Config

"""
PyPSANetworkAnalyzer: Python Class for Analyzing PyPSA Energy Networks
======================================================================

Overview:
---------
The `PyPSANetworkAnalyzer` class provides tools for analyzing and visualizing energy networks modeled using the PyPSA framework. It performs statistical analyses, generates insights into generator behaviors, and visualizes various network characteristics, such as dispatch profiles, marginal costs, and network layouts.

Key Features:
-------------
1. **Network Initialization**:
   - Loads a PyPSA network from the specified file.
   - Initializes network carriers and statistics.

2. **Analysis Functions**:
   - Calculates system costs, generator size changes, and line loadings.
   - Groups and visualizes generator statistics by carrier.

3. **Visualization Tools**:
   - Plots energy dispatch, network layouts, and cost breakdowns.
   - Customizable visualizations with options for scaling and coloring based on network attributes.

Class Attributes:
-----------------
- `network`:
  The loaded PyPSA network object.
- `results`:
  A dictionary storing analysis results such as generator statistics and line loading.
- `colors`:
  A mapping of carrier names to their respective colors for consistent visualizations.
- `statistics`:
  Basic statistics of the network.

"""


class PyPSANetworkAnalyzer:
    def __init__(self, network_file, config):
        """
        Initializes the class, loads the network, and sets up carriers and statistics.
        Parameters
        ----------
        network_file : str. Name of the network to upload
        config : object type config.

        Returns
        -------
        None.

        """
        # Verifica la directory corrente

        script_dir = Path(__file__).parent.resolve()
        root_dir = script_dir.parent
        
        if str(script_dir).split("\\")[-1] == 'scripts':
            network_path = f"{root_dir}/networks/{network_file}"
            self.output_folder = f"{root_dir}/results/{network_file}"
        else:
            network_path = f"networks/{network_file}"
            self.output_folder = f"results/{network_file}"
            

        # Carica il network
        self.network = pypsa.Network(network_path)
        self.config = config
        self.results = dict()
        
        # Creation of the output folder to save data
        
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        
        self.network.carriers = self.network.carriers.drop(self.network.carriers['color'].loc[''])
        if 'none' in self.network.carriers.index:
            self.network.carriers = self.network.carriers.drop('none')
        self.colors = self.network.carriers['color']
        
        self.statistics = self.network.statistics()
        
        self.init()
        
    def init(self):
        """
        Runs initial analyses and visualization workflows.

        Returns
        -------
        None.

        """
        
        self.analyze_network()
        
        if self.config.get('DISPATCH_BY_CARRIER', 'plot_time'):
            self.plot_dispatch()
        
        if self.config.get('NETWORK_PLOT', 'plot_network_pnomopt'):
            self.plot_network_p_nom_opt_gen()
            self.plot_network_p_nom_opt_stor()
        if self.config.get('NETWORK_PLOT', 'plot_network_marginalcost'):
            self.plot_network_marginal_cost()
        if self.config.get('NETWORK_PLOT', 'plot_network_totalload'):
            self.plot_network_total_load()    

    def analyze_network(self):
        """
        Performs high-level network analysis, including generator
        analysis and line loading calculations."""
        
        self.analyze_generators()
        self.analyze_lines()
        
        if self.config.get('SYSTEM_COST', 'plot'):
            self.system_cost()
        
        if self.config.get('NETWORK', 'nation_only_analysis'):
            nation = self.config.get('NETWORK', 'nation')
            self.results['total_load'] = self.network.loads_t.p.loc[:, self.network.loads_t.p.columns.str.startswith(nation)].sum() /1e3
        else:
            self.results['total_load'] = self.network.loads_t.p.sum() /1e3
        
        
        
    def analyze_generators(self):
        generators = self.network.generators
        generators_t = self.network.generators_t.p
        
        if self.config.get('NETWORK', 'nation_only_analysis'):
            nation = self.config.get('NETWORK', 'nation')
            generators = generators[generators.index.str.startswith(nation)]
            generators_t = generators_t.loc[:, generators_t.columns.str.startswith(nation)]
        
        generators_grouped = generators.groupby('carrier').sum()
        # In kW
        self.results['generators'] = pd.DataFrame((generators['p_nom_opt'] - generators['p_nom']), columns=['size_increase'])
       
        self.results['generators_grouped'] = pd.DataFrame(generators_grouped['p_nom_opt'], columns=['p_nom_opt'])
        self.results['generators_grouped_t'] = generators_t.T.groupby(generators['carrier']).sum().T
        self.results['generators_grouped']['increase_by_carrier'] = self.results['generators']['size_increase'].T.groupby(generators['carrier']).sum().T
        
        if self.config.get('DISPATCH_BY_CARRIER', 'plot_year'):
            self.plot_dispatchbycarrier_year(self.results['generators_grouped_t'])
            
        if self.config.get('DISPATCH_BY_CARRIER', 'plot_histogram_year'):
            self.plot_dispatchbycarrier_histogram(generators_t)
        
        if self.config.get('INCREASE_IN_SIZE_GENERATORS', 'plot'):
            self.plot_generators_increase(self.results['generators_grouped']['increase_by_carrier'])
            
        if self.config.get('SIZE_GENERATORS', 'plot'):
            self.plot_generators_size()
     
        
    def analyze_lines(self):
       lines = self.network.lines
       self.results['line_loading_max'] = self.network.lines_t.p0.abs().max(axis=0) / self.network.lines.s_nom_opt
       self.results['line_loading_mean'] = self.network.lines_t.p0.abs().mean(axis=0) / self.network.lines.s_nom_opt
       
       self.results['link_loading_max'] = self.network.links_t.p0.abs().max(axis=0) / self.network.links.p_nom_opt
       self.results['link_loading_mean'] = self.network.links_t.p0.abs().mean(axis=0) / self.network.links.p_nom_opt
       
       
       self.results['line_expansion'] = pd.DataFrame((lines['s_nom_opt'] - lines['s_nom']), columns=['line_expansion_absolute'])
       self.results['line_expansion']['line_expansion_relative'] = (lines['s_nom_opt'] - lines['s_nom']) / lines['s_nom']
        
    
    def plot_generators_increase(self, p_nom_opt):
        """Plot increase in size of generators as a bar chart with log scale."""
        
        offwind_sum = p_nom_opt.loc[["offwind-ac", "offwind-dc", "offwind-float"]].sum()
        p_nom_opt = p_nom_opt.drop(["offwind-ac", "offwind-dc", "offwind-float", 'geothermal', 'ror', 'solar-hsat'], errors='ignore')
        p_nom_opt.loc["offwind"] = offwind_sum
        
        colors = self.colors.copy()
        colors.loc['offwind'] = colors.loc['offwind-ac']
        
        colors = [colors.get(carrier, '#333333') for carrier in p_nom_opt.index]
        
        # Creazione del grafico
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = p_nom_opt.plot.bar(ax=ax, color=colors)
        
        # Aggiungere i valori sopra le barre
        for bar, value in zip(ax.patches, p_nom_opt.values):
            # Controlla che il valore non sia zero
            if value > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,  # Posizione orizzontale
                    bar.get_height(),  # Altezza della barra
                    f"{value:.2e}",  # Formattazione del valore
                    ha='center', va='bottom', fontsize=9, color='black'  # Allineamento e stile
                )
        
        # Personalizzazione del grafico
        ax.set_title("Increase in Size of Generators")
        ax.set_xlabel("Carrier")
        ax.set_ylabel("Size (MW)")
        ax.set_yscale("log")  # Imposta scala logaritmica per l'asse y
        ax.tick_params(axis='x', rotation=45)  # Ruota le etichette dell'asse x
        plt.tight_layout()
        
        if self.config.get('INCREASE_IN_SIZE_GENERATORS', 'save'):
            plt.savefig(f"{self.output_folder}/size_increase.png", format='png', dpi=300, bbox_inches='tight')
     
        
    def plot_dispatchbycarrier_histogram(self, generators_t):
        """Plot histogram of dispatch of carriers."""
        
        if self.config.get('NETWORK', 'nation_only_analysis'):
            
            italian_cols = generators_t.columns[generators_t.columns.str.startswith('IT')]
        
            carriers = self.network.generators.loc[italian_cols, 'carrier']
            nice_names = self.network.carriers.loc[carriers.values, 'nice_name'].values
        
            dispatch = (
                generators_t[italian_cols]
                .T.set_axis(nice_names)  # Imposta i carrier come index per il groupby
                .groupby(level=0)
                .sum()
                .sum(axis=1)
            )
        else:
            statistics = self.statistics.loc[~self.statistics.index.isin([('Load', '-'), ('Line', 'AC'), ('Link', 'DC')])].droplevel(0)
            dispatch = statistics['Supply'].T
            
        offwind_sum = dispatch.loc[["Offshore Wind (AC)", "Offshore Wind (DC)", "Offshore Wind (Floating)"]].sum()
        dispatch = dispatch.drop(["Offshore Wind (AC)", "Offshore Wind (DC)", "Offshore Wind (Floating)", 'geothermal', 'ror', 'solar-hsat'], errors='ignore')
        dispatch.loc["Offwind"] = offwind_sum
        
        # colors = [colors.get(carrier, '#333333') for carrier in dispatch.index]
        
        # Extract the color for each carrier from the network.carriers dataframe
        carrier_colors = self.network.carriers.set_index('nice_name')['color']
        
        # Map the dispatch index (which are nice names) to their corresponding color
        colors = dispatch.index.to_series().map(carrier_colors).fillna('#333333')
        
        # Optionally overwrite specific colors (e.g., custom color for Offwind)
        colors.loc["Offwind"] = "#6895dd"

        
        # Creazione del grafico
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = dispatch.plot.bar(ax=ax, color=colors)
        
        # Aggiungere i valori sopra le barre
        for bar, value in zip(ax.patches, dispatch.values):
            # Controlla che il valore non sia zero
            if value > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,  # Posizione orizzontale
                    bar.get_height(),  # Altezza della barra
                    f"{value:.2e}",  # Formattazione del valore
                    ha='center', va='bottom', fontsize=7, color='black'  # Allineamento e stile
                )
        
        # Personalizzazione del grafico
        ax.set_title("Supply for generators")
        ax.set_xlabel("Carrier")
        ax.set_ylabel("Size (MW)")
        ax.set_yscale("log")  # Imposta scala logaritmica per l'asse y
        ax.tick_params(axis='x', rotation=90)  # Ruota le etichette dell'asse x
        plt.tight_layout()
        
        if self.config.get('DISPATCH_BY_CARRIER', 'save_histogram_year'):
            plt.savefig(f"{self.output_folder}/supply_year.png", format='png', dpi=300, bbox_inches='tight')
            
        
    
    
    def plot_generators_size(self):
        
        line_technologies = ['AC', 'DC']
        
        if self.config.get('NETWORK', 'nation_only_analysis'):
            nation = self.config.get('NETWORK', 'nation')
            generators = self.network.generators[self.network.generators.index.str.startswith(nation)]
            storage_units = self.network.storage_units[self.network.storage_units.index.str.startswith(nation)]
            stores = self.network.stores[self.network.stores.index.str.startswith(nation)] if not self.network.stores.empty else self.network.stores
            links = self.network.links[self.network.links.index.str.startswith(nation)]
            lines = self.network.lines[self.network.lines.bus0.str.startswith(nation) | self.network.lines.bus1.str.startswith('IT')]
            
            optimal_capacity = pd.concat([
                generators.groupby('carrier').sum().p_nom_opt,
                storage_units.groupby('carrier').sum().p_nom_opt,
                stores.groupby('carrier').sum().e_nom_opt,
                links.groupby('carrier').sum().p_nom_opt,
                lines.groupby('carrier').sum().s_nom_opt
            ])
            
            installed_capacity = pd.concat([
                generators.groupby('carrier').sum().p_nom,
                storage_units.groupby('carrier').sum().p_nom,
                stores.groupby('carrier').sum().e_nom,
                links.groupby('carrier').sum().p_nom,
                lines.groupby('carrier').sum().s_nom
            ])
            
            optimal_capacity.index = [self.network.carriers.loc[index]['nice_name'] for index in optimal_capacity.index]
            installed_capacity.index = [self.network.carriers.loc[index]['nice_name'] for index in installed_capacity.index]
        else:
            statistics = self.statistics.loc[self.statistics.index != ('Load', '-')].droplevel(0)
        
            # Sposta "AC" e "DC" alla fine
            main_statistics = statistics.loc[~statistics.index.isin(line_technologies)]
            line_statistics = statistics.loc[statistics.index.isin(line_technologies)] /1e3
            statistics = pd.concat([main_statistics, line_statistics])  # Ricombina, con AC/DC alla fine
            
            optimal_capacity = statistics['Optimal Capacity']
            installed_capacity = statistics['Installed Capacity']
    
        # Creazione della figura
        technologies = optimal_capacity.index.tolist()
        x = np.arange(len(technologies))  # Posizioni sull'asse x
        width = 0.35  # Larghezza delle barre
    
        fig, ax1 = plt.subplots(figsize=(14, 8))
    
        # Istogramma per le tecnologie principali
        bars1 = ax1.bar(x - width/2, optimal_capacity, width, label='Optimal Capacity', color='skyblue')
        bars2 = ax1.bar(x + width/2, installed_capacity, width, label='Installed Capacity', color='salmon')
    
        # Aggiunta dei valori sopra le barre solo per Optimal Capacity
        for (i,bar) in enumerate(bars1):
            height = bar.get_height()
            if height > 0 and i<len(bars1)-2:  # Mostra solo se il valore è maggiore di zero
                ax1.text(bar.get_x() + bar.get_width() / 2, height + max(optimal_capacity) * 0.01,
                         f'{height:.2e}', ha='center', va='bottom', fontsize=5)
    
        # Scala secondaria per le tecnologie AC e DC
        ax2 = ax1.twinx()  # Secondo asse y condiviso
        line_indices = [i for i, tech in enumerate(technologies) if tech in line_technologies]
    
        # Aggiunta delle barre per AC e DC nell'asse secondario
        bars_ac_dc = []
        for idx in line_indices:
            bar = ax2.bar(x[idx] - width / 2, optimal_capacity.iloc[idx], width, color='lightgreen',
                          label='Optimal Capacity (AC/DC) [GW]' if not bars_ac_dc else "")
            for barr in bar:
                height = barr.get_height()
                if height > 0:  # Mostra solo se il valore è maggiore di zero
                    ax2.text(barr.get_x() + barr.get_width() / 2, height ,
                             f'{height:.2e}', ha='center', va='bottom', fontsize=5)
            bars_ac_dc.append(bar)
    
        # Configurazione del primo asse
        ax1.set_title('Confronto delle tecnologie con scala separata per AC/DC', fontsize=16)
        ax1.set_xlabel('Tecnologie', fontsize=12)
        ax1.set_ylabel('Capacità (tecnologie principali)', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(technologies, rotation=45, ha='right', fontsize=10)
        ax1.legend(loc='upper left')
    
        # Configurazione del secondo asse
        ax2.set_ylabel('Capacità (AC/DC)', fontsize=12)
    
        # Layout compatto
        plt.tight_layout()
    
        # Salvataggio dell'immagine
        if self.config.get('SIZE_GENERATORS', 'save'):
            plt.savefig(f"{self.output_folder}/size_generators.png", format='png', dpi=300, bbox_inches='tight')



    def plot_dispatchbycarrier_year(self, dispatch_by_carrier):
        """
        Plots aggregated dispatch data by carrier over time.

        Parameters
        ----------
        dispatch_by_carrier : TYPE
            DESCRIPTION.
        colors : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        colors = [self.colors.get(carrier, '#333333') for carrier in dispatch_by_carrier.columns]
        dispatch_by_carrier.plot(figsize=(10, 6), color=colors)
        plt.title("Dispatch by Carrier Over Time")
        plt.xlabel("Time")
        plt.ylabel("Dispatch (MW)")
        plt.legend(title="Carrier")
        plt.tight_layout()
        
        if self.config.get('DISPATCH_BY_CARRIER', 'save_year'):
            plt.savefig(f"{self.output_folder}/dispatch_year.png", format='png', dpi=300, bbox_inches='tight')
        

    def plot_dispatch(self):
        """
        Visualizes the energy balance for a specified day.

        Parameters
        ----------
        n : PyPSA object
            PyPSA network under analysis
        time : str.
            Time period to be plotted
        ylim: list.
            Limit for the y-axis of the graph

        Returns
        -------
        None.

        """
        
        n = self.network
        _time = self.config.get('DISPATCH_BY_CARRIER', 'time')
        time = f"{self.network.snapshots[0].year}-{_time}"
        ylim = self.config.get('DISPATCH_BY_CARRIER','ylim')
        
        
        p = (
            n.statistics.energy_balance(aggregate_time=False)
            .groupby("carrier")
            .sum()
            .div(1e3)
            .T
        )
        
        if '-' in p.columns:
            p = p.drop('-', axis=1)
    
        fig, ax = plt.subplots(figsize=(6, 3))
        color = p.columns.map(n.carriers.set_index('nice_name').color)
    
        p.where(p > 0).loc[time].plot.area(
            ax=ax,
            linewidth=0,
            color=color,
            title = f'Energy balance for day {time}'
        )
    
        charge = p.where(p < 0).dropna(how="all", axis=1).loc[time]
    
        if not charge.empty:
            charge.plot.area(
                ax=ax,
                linewidth=0,
                color=charge.columns.map(n.carriers.set_index('nice_name').color),
            )
    
        n.loads_t.p_set.sum(axis=1).loc[time].div(1e3).plot(ax=ax, c="k")
    
        plt.legend(loc=(1.05, 0), fontsize=5)
        ax.set_ylabel("GW")
        
        ax.set_ylim(ylim[0], ylim[1])
        
        
        if self.config.get('DISPATCH_BY_CARRIER', 'save_time'):
            plt.savefig(f"{self.output_folder}/dispatch_{_time}.png", format='png', dpi=300, bbox_inches='tight')
        

    def system_cost(self):
        """
        Calculates and visualizes the total system cost breakdown by technology.
    
        Parameters
        ----------
        None
    
        Returns
        -------
        None.
        """
        tsc = pd.concat([self.network.statistics.capex(), self.network.statistics.opex()], axis=1)
        system_cost = tsc.sum(axis=1).droplevel(0).div(1e9).round(2)  # billion €/a
    
        fig, ax = plt.subplots(figsize=(6, 3))
    
        # Convert to a list for Matplotlib's plt.pie()
        labels = system_cost.index
        values = system_cost.values
    
        # Define a function to display only non-zero percentages and format them smaller
        def autopct(pct):
            return f'{pct:.1f}%' if pct > 0.1 else ''
    
        # Create the pie chart
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct=autopct,  # Apply the custom autopct function
            textprops={'fontsize': 3},  # Reduce font size for labels
            startangle=90  # Rotate for better alignment
        )
    
        # Adjust autotexts (percentages) size for better readability
        for autotext in autotexts:
            autotext.set_fontsize(5)  # Make percentages smaller
            autotext.set_color('black')  # Ensure readability
    
        # Save the plot if specified in the config
        if self.config.get('SYSTEM_COST', 'save'):
            plt.savefig(f"{self.output_folder}/system_cost_pie.png", format='png', dpi=300, bbox_inches='tight')
    
        plt.title('Total cost per technology')
        plt.tight_layout()  # Adjust layout for better fit
        plt.show()


        
    def plot_network_p_nom_opt_gen(self):
        """Plot the network layout, based on the optimal size of the generators (s)"""
        
        # Calculate the size of the buses based on the optimal size of the generators
        s = self.network.generators.p_nom_opt.groupby([self.network.generators.bus, self.network.generators.carrier]).sum()
        p_nom_opt = self.network.generators.p_nom_opt.groupby(self.network.generators.carrier).sum()
        colors = [self.colors.get(carrier, '#333333') for carrier in p_nom_opt.index]
        
        title = "Network Layout per generators optimal capacity"
        output = 'generators'
        
        self.plot_nom_opt(s, p_nom_opt, colors, title, output)
        
    def plot_network_p_nom_opt_stor(self):
        """Plot the network layout, based on the optimal size of the generators (s)"""
        
        # 1. Concatenare p_nom_opt da generators, storage_units e stores
        all_pnoms = pd.concat([
            self.network.storage_units.assign(component='storage_unit')[['bus', 'carrier', 'p_nom_opt']],
            self.network.stores.assign(component='store')[['bus', 'carrier', 'e_nom_opt']].rename(columns={'e_nom_opt': 'p_nom_opt'})
        ])
        
        # 2. Calcolare le dimensioni dei bus (gruppati per bus e carrier)
        s = all_pnoms.groupby(['bus', 'carrier'])['p_nom_opt'].sum()
        
        # 3. Somma totale per ogni carrier (per la legenda)
        p_nom_opt = all_pnoms.groupby('carrier')['p_nom_opt'].sum()
        
        # 4. Colori da dizionario self.colors
        colors = [self.colors.get(carrier, '#333333') for carrier in p_nom_opt.index]
        
        title = "Network Layout per storages optimal capacity"
        output = 'stores'

        
        self.plot_nom_opt(s, p_nom_opt, colors, title, output)
        
        
    def plot_nom_opt(self, s, p_nom_opt, colors, title, output):
        
        fig = plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.OCEAN, color="azure")
        ax.add_feature(cartopy.feature.LAND, color="cornsilk")
        
        self.network.plot(ax=ax,
                          margin=0.1,
                          bus_sizes=s / self.config.get('NETWORK_PLOT', 'bus_scaling_factor'),
                          branch_components = ["Line", "Link"],
                          line_widths=self.network.lines.s_nom_opt/4e3,
                          link_widths=self.network.links.p_nom_opt/4e3,
                          line_colors = self.results['line_loading_mean'],
                          link_colors = self.results['link_loading_mean'],
                          line_cmap = plt.cm.viridis, 
                          )
        

        
        # Add legend based on carriers
        add_legend_patches(
            ax=ax,
            colors=colors,
            labels=p_nom_opt.index,
            legend_kw=dict(frameon=True,
                           loc='upper right', fontsize=6,
                           title='Carriers', title_fontsize=6)
        )
        
        # Parametri per la legenda delle linee
        cmap = plt.cm.viridis
        norm = plt.Normalize(
            vmin=self.results['line_loading_mean'].min(),
            vmax=self.results['line_loading_mean'].max()
        )
        
        # Numero di etichette nella legenda
        n_labels = 6
        values = [norm.vmin + i * (norm.vmax - norm.vmin) / (n_labels - 1) for i in range(n_labels)]
        labels = [f"{v:.2f}" for v in values]
        handles = [mpl.patches.Patch(color=cmap(norm(v))) for v in values]
        
        # Crea la seconda legenda
        line_legend = ax.legend(
            handles, labels, title="Line loading",
            loc="lower left", frameon=True, fontsize=6,
            title_fontsize=6
        )
        
        # Aggiungi la seconda legenda al grafico
        ax.add_artist(line_legend)
        
        # Add title
        plt.title(title, fontweight='bold')
        plt.tight_layout()
        
        if self.config.get('NETWORK_PLOT', 'save_network_pnomopt'):
            plt.savefig(f"{self.output_folder}/network_pnomopt_{output}.png", dpi=300, bbox_inches='tight')
    
    
    def plot_network_total_load(self):
        """Plot the network layout, based on the total load per bus (s)"""
        
        # Calculate the size of the buses based on the optimal size of the generators
        s = self.network.loads_t.p.sum() / 5e3
        fig = plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.OCEAN, color="azure")
        ax.add_feature(cartopy.feature.LAND, color="cornsilk")
        
        if self.config.get('NETWORK', 'sector_coupled'):
            self.network.plot(ax=ax, margin=0.1,
                              line_widths=self.results['line_loading_mean']*self.config.get('NETWORK_PLOT', 'line_scaling_factor'),
                              )
        else:
            self.network.plot(ax=ax, margin=0.1, bus_sizes=s / self.config.get('NETWORK_PLOT', 'bus_scaling_factor'),
                              line_widths=self.results['line_loading_mean']*self.config.get('NETWORK_PLOT', 'line_scaling_factor'),
                              )
        
        # Add title
        plt.title("Network Layout per total demand")
        plt.tight_layout()
        # plt.savefig(f"{output_folder}/network_layout.png")
        
        if self.config.get('NETWORK_PLOT', 'save_network_totalload'):
            plt.savefig(f"{self.output_folder}/network_totalload.png", dpi=300, bbox_inches='tight')
        
     
    def plot_network_marginal_cost(self):
        """Plot the network layout, based on the marginal cost per buses"""
        
        cols = self.network.buses_t.marginal_price.columns
        filtered_cols = cols[
            cols.str.match(r'^.*\d+$') & ~cols.str.contains("battery|H2", case=False)
        ]
        filtered = self.network.buses_t.marginal_price[filtered_cols]
        
        fig = plt.figure(figsize=(7, 7))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.OCEAN, color="azure")
        ax.add_feature(cartopy.feature.LAND, color="cornsilk")
        
        vnorm = self.config.get('NETWORK_PLOT', 'range_normalization')
        norm = plt.Normalize(vmin=vnorm[0], vmax=vnorm[1])  # €/MWh
        
        self.network.plot(
            ax=ax,
            bus_colors=filtered.mean(),
            bus_cmap="plasma",
            bus_norm=norm,
            bus_alpha=1,
            bus_sizes=0.1
        )
        
        plt.colorbar(
            plt.cm.ScalarMappable(cmap="plasma", norm=norm),
            ax=ax,
            label="LMP [€/MWh]",
            shrink=0.6,
        )
    
        plt.title("Network marginal cost per bus")
        
        if self.config.get('NETWORK_PLOT', 'save_network_marginalcost'):
            plt.savefig(f"{self.output_folder}/network_marginalcost.png", dpi=300, bbox_inches='tight')

# Usage example
if __name__ == "__main__":
    config = Config()
    network_name = '2040_deit.nc'
    network_analyzer =  PyPSANetworkAnalyzer(network_name, config)



