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
from matplotlib.colors import to_rgba
import matplotlib.patches as mpatches

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
            self.plot_network_p_nom_opt_all()
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
        ax.set_ylabel("Size (MW)", fontsize = 14, fontweight = "bold")
        ax.set_title("Increase in Size of Generators", fontsize=16, fontweight='bold')


        # Assegna le etichette formattate all'asse X
        x = np.arange(len(p_nom_opt.index))
        ax.set_xticks(x)
        ax.set_xticklabels(p_nom_opt.index, rotation=30, ha="right", fontsize=10, fontstyle='italic', color='#2F4F4F')  # Rotazione a 30°
        
        ax.set_yscale("log")  # Imposta scala logaritmica per l'asse y
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
        
        dispatch = dispatch / 1e3 # TWh
        
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
        
        # Personalizzazione del graficoù
        ax.set_ylabel("Energy (TWh)", fontsize = 14, fontweight = "bold")
        ax.set_title("Total energy dispatch by carrier", fontsize=16, fontweight='bold')


        # Assegna le etichette formattate all'asse X
        x = np.arange(len(dispatch.index))
        ax.set_xticks(x)
        ax.set_xticklabels(dispatch.index, rotation=30, ha="right", fontsize=10, fontstyle='italic', color='#2F4F4F')  # Rotazione a 30°
        
        
        ax.set_yscale("log")  # Imposta scala logaritmica per l'asse y
        plt.tight_layout()
        
        if self.config.get('DISPATCH_BY_CARRIER', 'save_histogram_year'):
            plt.savefig(f"{self.output_folder}/supply_year.png", format='png', dpi=300, bbox_inches='tight')
            
        
    

    def plot_generators_size(self):
        if self.config.get('NETWORK', 'nation_only_analysis'):
            nation = self.config.get('NETWORK', 'nation')
            generators = self.network.generators[self.network.generators.index.str.startswith(nation)]
            storage_units = self.network.storage_units[self.network.storage_units.index.str.startswith(nation)]
            stores = self.network.stores[self.network.stores.index.str.startswith(nation)] if not self.network.stores.empty else self.network.stores
    
            # Build optimal capacity Series
            optimal_capacity = pd.concat([
                generators.groupby('carrier').sum().p_nom_opt,
                storage_units.groupby('carrier').sum().p_nom_opt,
                stores.groupby('carrier').sum().e_nom_opt,
            ])
        else:
            statistics = self.statistics.loc[self.statistics.index != ('Load', '-')].droplevel(0)
            optimal_capacity = statistics['Optimal Capacity']
    
        # Group offwind
        offwind_keys = ['offwind-ac', 'offwind-dc', 'offwind-float']
        offwind_sum = optimal_capacity.get(offwind_keys, pd.Series()).sum()
        if offwind_sum > 0:
            optimal_capacity.loc['offwind'] = offwind_sum
        optimal_capacity = optimal_capacity.drop(offwind_keys, errors='ignore')
    
        # Group hydro components
        hydro_keys = ['ror', 'PHS', 'hydro']
        hydro_sum = optimal_capacity.get(hydro_keys, pd.Series()).sum()
        if hydro_sum > 0:
            optimal_capacity.loc['Hydro'] = hydro_sum
        optimal_capacity = optimal_capacity.drop(hydro_keys, errors='ignore')
    
        # Drop battery charger/discharger
        optimal_capacity = optimal_capacity.drop(['battery charger', 'battery discharger'], errors='ignore')
    
        # Filter by minimum capacity
        optimal_capacity = optimal_capacity[optimal_capacity >= 1]  # ≥ 1 MW
    
        technologies = optimal_capacity.index.tolist()
        x = np.arange(len(technologies))
        width = 0.6
    
        # Color mapping
        base_colors = self.colors.copy()
        base_colors.loc['offwind'] = base_colors.get('offwind-ac', '#333333')
        base_colors.loc['Hydro'] = base_colors.get('hydro', '#0072B2')
        color_list = [base_colors.get(tech, '#333333') for tech in technologies]
    
        # Plotting
        fig, ax = plt.subplots(figsize=(14, 8))
        bars = ax.bar(x, optimal_capacity, width, label='Optimal Capacity', color=color_list)
    
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, height + max(optimal_capacity) * 0.01,
                        f'{height:.2e}', ha='center', va='bottom', fontsize=9)
    
        ax.set_title('Optimal Capacity per Technology', fontsize=16, fontweight='bold')
        ax.set_xlabel('Technologies', fontsize=12, fontweight='bold')
        ax.set_ylabel('Capacity [kW]', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(technologies, rotation=45, ha='right', fontsize=10)
        ax.legend()
    
        plt.tight_layout()
    
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
        """Plot network layout showing storage capacities grouped by location."""
    
        # Combine all storages into a single dataframe
        all_pnoms = pd.concat([
            self.network.storage_units.assign(component='storage_unit')[['bus', 'carrier', 'p_nom_opt']],
            self.network.stores.assign(component='store')[['bus', 'carrier', 'e_nom_opt']].rename(columns={'e_nom_opt': 'p_nom_opt'})
        ])
    
        # Map secondary buses (battery/H2) to their parent bus (e.g., 'AT0 0 battery' → 'AT0 0')
        def get_main_bus(bus):
            if bus.endswith(" battery") or bus.endswith(" H2"):
                return " ".join(bus.split(" ")[:-1])
            return bus
    
        all_pnoms['main_bus'] = all_pnoms['bus'].apply(get_main_bus)
    
        # Re-aggregate by main_bus and carrier
        s = all_pnoms.groupby(['main_bus', 'carrier'])['p_nom_opt'].sum()
    
        # Total capacity per carrier (used for legend)
        p_nom_opt = all_pnoms.groupby('carrier')['p_nom_opt'].sum()
    
        # Assign colors
        colors = [self.colors.get(carrier, '#333333') for carrier in p_nom_opt.index]
    
        # Call general plotting function
        title = "Network Layout per storage units optimal capacity"
        output = 'stores'
    
        self.plot_nom_opt(s, p_nom_opt, colors, title, output)
        
    
    def plot_network_p_nom_opt_all(self):
        """Plot network layout combining generators, storage units, and stores."""
    
        # Combine all components into one DataFrame
        gen_df = self.network.generators[['bus', 'carrier', 'p_nom_opt']].copy()
        su_df = self.network.storage_units[['bus', 'carrier', 'p_nom_opt']].copy()
        sto_df = self.network.stores[['bus', 'carrier', 'e_nom_opt']].rename(columns={'e_nom_opt': 'p_nom_opt'}).copy()
    
        # Identify the component type (optional, could be useful later)
        gen_df['component'] = 'generator'
        su_df['component'] = 'storage_unit'
        sto_df['component'] = 'store'
    
        # Concatenate all components
        all_pnoms = pd.concat([gen_df, su_df, sto_df], ignore_index=True)
    
        # Normalize bus names for H2 and battery suffixes
        def get_main_bus(bus):
            if bus.endswith(" battery") or bus.endswith(" H2"):
                return " ".join(bus.split(" ")[:-1])
            return bus
    
        all_pnoms['main_bus'] = all_pnoms['bus'].apply(get_main_bus)
    
        # Group by bus and carrier to get sizes
        s = all_pnoms.groupby(['main_bus', 'carrier'])['p_nom_opt'].sum()
    
        # Total capacity per carrier for the legend
        p_nom_opt = all_pnoms.groupby('carrier')['p_nom_opt'].sum()
    
        # Assign colors
        colors = [self.colors.get(carrier, '#333333') for carrier in p_nom_opt.index]
    
        # Call the generic plotting function
        title = "Network Layout per total optimal capacity"
        output = 'all_components'
    
        self.plot_nom_opt(s, p_nom_opt, colors, title, output)

        
        
    def plot_nom_opt(self, s, p_nom_opt, colors, title, output):
        
        
        legend_map = {
            'CCGT': 'Combined-cycle gas',
            'biomass': 'Biomass',
            'geothermal': 'Geothermal',
            'nuclear': 'Nuclear',
            'offwind-ac': 'Offwind',
            'offwind-dc': 'Offwind',
            'offwind-float': 'Offwind',
            'oil': 'Oil',
            'onwind': 'Wind onshore',
            'ror': 'Run of river',
            'solar': 'Photovoltaic',
            'solar-hsat': 'solar hsat',
            'H2': 'Hydrogen',
            'PHS': 'Pumped-hydro storage',
            'battery': 'Battery storage',
            'hydro': 'Hydro storage'
            }
            
        
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
        
        # Plot only bus names that do NOT end with 'battery' or 'H2'
        if self.config.get('NETWORK_PLOT', 'show_bus_labels'):
            for bus_name, row in self.network.buses.iterrows():
                if not (bus_name.lower().endswith('battery') or bus_name.lower().endswith('h2')):
                    ax.text(row['x'], row['y'], bus_name, fontsize=4, ha='center', va='center',
                            transform=ccrs.PlateCarree(), zorder=5)


        
        # Filter out entries with capacity <= 1
        legend_mask = p_nom_opt > 1
        filtered_labels = p_nom_opt[legend_mask].index
        filtered_colors = [colors[i] for i, label in enumerate(p_nom_opt.index) if label in filtered_labels]
        display_labels = [legend_map.get(label, label) for label in filtered_labels]  # Fallback to original if not mapped
                
        add_legend_patches(
            ax=ax,
            colors=filtered_colors,
            labels=display_labels,
            legend_kw=dict(frameon=True,
                           loc='upper right', fontsize=5,
                           title='Carriers', title_fontsize=6, framealpha=0.8)
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
            title_fontsize=6, framealpha=0.5
        )
        
        # Aggiungi la seconda legenda al grafico
        ax.add_artist(line_legend)
        
        bus_scale = self.config.get('NETWORK_PLOT', 'bus_scaling_factor')
        ref_values = [3e6, 1e7]  # in kW ad esempio (1 GW, 5 GW)
        ref_sizes = [v / bus_scale for v in ref_values]  # come nel plot

        ref_colors = ['#1f77b4', '#1f77b4']
        
        size_handles = [
            plt.scatter([], [], s=size,
                        facecolors=color, alpha=0.6)
            for size, color in zip(ref_sizes, ref_colors)
        ]
        size_labels = [f"{v / 1e6:.1f} GW" for v in ref_values]

        
        size_legend = ax.legend(
            size_handles, size_labels,
            loc='upper left',
            frameon=True,
            fontsize=6,
            title_fontsize=6,
            framealpha=0.3,
            handletextpad=2,
            labelspacing=1.2,
            borderaxespad=0
            ,
            borderpad = 1.2
        )
        ax.add_artist(size_legend)
        
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
            bus_sizes=0.1,
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
    network_name = '2040_deit_geothermal.nc'
    network_analyzer =  PyPSANetworkAnalyzer(network_name, config)



