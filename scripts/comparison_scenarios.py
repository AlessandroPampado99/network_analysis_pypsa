import os
import pypsa
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class ScenarioComparison:
    
    def __init__(self, n_list, nation='IT'):
        self.n_list = n_list
        self.nation = nation
        self.output_folder = f"../results/comparison_scenarios"
        
        # Dictionary for scenario conversion
        self.conversion_scenario = {
            "2040_deit_def": "DE-IT",
            "2040_deit_biomass_limit": "DE-IT",
            "commodity_nze": "WEO",
            "low_electrical_demand": "LED",
            "nuclear_base": "NCL",
            "no_co2_emissions": "NOCO2",
            'nuclear_no_emissions': "NCLCO2",
            'no_biomass_limit': 'BIO',
            'prova_nuclear_installazione': "PROVA"
        }
        
        # Dictionary to map technology short names to full descriptions
        self.legend_map = {
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
            'solar-hsat': 'Solar hsat',
            'H2': 'Hydrogen',
            'PHS': 'Pumped-hydro storage',
            'battery': 'Battery storage',
            'Hydro': 'Hydro storage',
            
        }
        
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        self.add_network()

    def add_network(self):
        # Initialize a dictionary to store p_nom_opt data for each scenario
        p_nom_opt_data = {}
        p_dispatch_data = {}
        objective_data = {}
        capacity_factor_data = {}
        
        
        for network in self.n_list:
            n = pypsa.Network(f"../networks/{network}.nc")
            
            # Calculate p_nom_opt for the network
            p_nom_opt_data[network] = self.p_nom_opt_evaluation(n)
            p_dispatch_data[network] = self.p_dispatch_evaluation(n)
            objective_data[network] = self.objective_evaluation(n, self.nation)
            objective_data[network] = self.marginal_prices_evaluation(n, self.nation, objective_data[network])
            capacity_factor_data[network] = self.capacity_factor_evaluation(n)
        
        # self.assert_objective(n)
        
        # Now create the DataFrame that will be passed to the plot function
        p_nom_opt_data = pd.DataFrame(p_nom_opt_data)
        p_dispatch_data = pd.DataFrame(p_dispatch_data)
        objective_data = pd.DataFrame(objective_data, index=['objective', 'average_price'])
        capacity_factor_data = pd.DataFrame(capacity_factor_data)
        
        p_nom_opt_data, p_dispatch_data = self.clean_hydrogen(p_nom_opt_data, p_dispatch_data)
        
        # Plot the bar chart to visualize the results
        self.plot_bar_chart(p_nom_opt_data, 'Optimal Nominal Power for Technologies and Scenarios', 'Technology', 'Optimal size (GW)', 'optimal_size')
        self.plot_bar_chart(p_dispatch_data, 'Total dispatch for Technologies and Scenarios', 'Technology', 'Energy (TWh)', 'total_dispatch')
        self.plot_bar_chart(capacity_factor_data, 'Capacity factor for Technologies and Scenarios', 'Technology', 'Capacity factor', 'capacity_factor')
        self.plot_objective_with_avg_price(objective_data, 'Total annualised cost and average electricity cost per scenario', 'Scenario', 'Total cost (B€/year)', 'Average cost (€/MWh)', 'Objective_average_prices')
        
    def p_nom_opt_evaluation(self, network):
        """
        Calculate and return the p_nom_opt value for the network.
        It is assumed that the network contains information for each technology.
        """
        
        nation = self.nation
        generators = network.generators[network.generators.index.str.startswith(nation)]
        storage_units = network.storage_units[network.storage_units.index.str.startswith(nation)]
        stores = network.stores[network.stores.index.str.startswith(nation)] if not network.stores.empty else network.stores
    
        # Build optimal capacity Series
        optimal_capacity = pd.concat([
            generators.groupby('carrier').sum().p_nom_opt,
            storage_units.groupby('carrier').sum().p_nom_opt,
            stores.groupby('carrier').sum().e_nom_opt,
        ])
    
        optimal_capacity = self.clean_carriers_series(optimal_capacity)
        
        optimal_capacity = optimal_capacity / 1e3 #GW
        
        
        return optimal_capacity
    
    def capacity_factor_evaluation(self, network):
        """
        Compute the average capacity factor per technology for a given nation's components,
        including generators, storage units, and stores.
    
        Returns
        -------
        pd.Series
            Capacity factor per technology (unitless, between 0 and 1).
        """
        nation = self.nation
        hours_in_year = 8760
    
        # Select components by nation
        generators = network.generators[network.generators.index.str.startswith(nation)]
        storage_units = network.storage_units[network.storage_units.index.str.startswith(nation)]
        stores = network.stores[network.stores.index.str.startswith(nation)]
    
        # Get power/electricity time series
        gen_p = network.generators_t.p[generators.index]
        su_p = network.storage_units_t.p[storage_units.index].abs()
        sto_e = network.stores_t.e[stores.index].abs() if not stores.empty else pd.DataFrame()
    
        # Total dispatched energy [MWh]
        gen_energy = gen_p.sum(axis=0)
        su_energy = su_p.sum(axis=0)
        sto_energy = sto_e.sum(axis=0) if not sto_e.empty else pd.Series(dtype=float)
    
        # Concatenate energy data
        all_energy = pd.concat([gen_energy, su_energy, sto_energy])
    
        # Carriers for grouping
        carriers = pd.concat([
            generators.loc[gen_energy.index, 'carrier'],
            storage_units.loc[su_energy.index, 'carrier'],
            stores.loc[sto_energy.index, 'carrier'] if not sto_energy.empty else pd.Series(dtype=str)
        ])
    
        total_energy_by_carrier = all_energy.T.groupby(carriers).sum()
    
        # Nominal capacity [MW or MWh]
        gen_nom = generators.groupby('carrier').sum().p_nom_opt
        su_nom = storage_units.groupby('carrier').sum().p_nom_opt
        sto_nom = stores.groupby('carrier').sum().e_nom_opt if not stores.empty else pd.Series(dtype=float)
    
        total_nominal_by_carrier = pd.concat([gen_nom, su_nom, sto_nom]).groupby(level=0).sum()
    
        # Compute capacity factor
        capacity_factor = total_energy_by_carrier / (total_nominal_by_carrier * hours_in_year) * 100
    
        # Clean and group carriers
        capacity_factor = self.clean_carriers_series(capacity_factor)
    
        return capacity_factor

    
    def p_dispatch_evaluation(self, network):
        """
        Calculate and return the dispatched active power (p_dispatch) for each carrier in the network.
        It is assumed that the network contains information for each technology (generators, storage units, stores).
        The calculation is done only for Italy.
        """
        
        nation = self.nation
        
        # Get dispatched active power for generators (based on the nation)
        generators_t = network.generators_t.p.loc[:, network.generators_t.p.columns.str.startswith(nation)]
        
        # Identify the relevant columns for Italian generators
        italian_cols = generators_t.columns
        
        # Get the carriers associated with the generators in the Italian columns
        carriers = network.generators.loc[italian_cols, 'carrier']
        
        # Sum the dispatched power per carrier for generators
        dispatched_power = generators_t.groupby(carriers, axis=1).sum()
        
        # Get dispatched active power for storage units (use p_dispatch)
        storage_units_t = network.storage_units_t.p.loc[:, network.storage_units_t.p_dispatch.columns.str.startswith(nation)]
        storage_carriers = network.storage_units.loc[storage_units_t.columns, 'carrier']
        
        # Sum the dispatched power per carrier for storage units
        storage_dispatched_power = storage_units_t.groupby(storage_carriers, axis=1).sum()
        
        # Get dispatched active power for stores (if available), sum only positive p values
        if not network.stores.empty:
            stores_t = network.stores_t.e.loc[:, network.stores_t.e.columns.str.startswith(nation)]
            store_carriers = network.stores.loc[stores_t.columns, 'carrier']
            
            # Only sum p values greater than 0 (energy provided, not absorbed)
            stores_dispatched_power = stores_t.groupby(store_carriers, axis=1).sum()
        else:
            stores_dispatched_power = pd.DataFrame()
        
        # Concatenate all power data into one DataFrame
        total_dispatched_power = pd.concat([dispatched_power, storage_dispatched_power, stores_dispatched_power], axis=1)
        
        # Sum the power for each carrier across all types of assets (generators, storage, stores)
        total_dispatched_power_per_carrier = total_dispatched_power.sum(axis=0) / 1e6 # TWh
        
        total_dispatched_power_per_carrier = self.clean_carriers_series(total_dispatched_power_per_carrier)
    
        return total_dispatched_power_per_carrier

    def clean_carriers_series(self, series: pd.Series) -> pd.Series:
        """
        Clean and group a pandas Series containing values by carrier.
        
        - Group offshore wind technologies into 'offwind'
        - Group 'ror', 'PHS', 'hydro' into 'Hydro'
        - Drop 'battery charger' and 'battery discharger'
        - Drop values < 1e-3 (1 MW if unit is kW, or 1 GWh if unit is MWh)
        - Ensure 'nuclear' is present
        
        Parameters
        ----------
        series : pd.Series
            Input series with carrier names as index and values (e.g., capacity or dispatch).
        
        Returns
        -------
        pd.Series
            Cleaned and grouped series.
        """
        series = series.copy()
    
        # Combine offshore wind into 'offwind'
        offwind_keys = ['offwind-ac', 'offwind-dc', 'offwind-float']
        offwind_sum = series.get(offwind_keys, pd.Series()).sum()
        if offwind_sum > 0:
            series.loc['offwind'] = offwind_sum
        series = series.drop(offwind_keys, errors='ignore')
    
        # Combine hydro types into 'Hydro'
        hydro_keys = ['ror', 'PHS', 'hydro']
        hydro_sum = series.get(hydro_keys, pd.Series()).sum()
        if hydro_sum > 0:
            series.loc['Hydro'] = hydro_sum
        series = series.drop(hydro_keys, errors='ignore')
    
        # Remove unwanted carriers
        to_remove = ['battery charger', 'battery discharger']
        series = series.drop(to_remove, errors='ignore')
        
        # Filter by threshold ≥ 1e-1
        series = series[series >= 5e-2]
    
        # Ensure 'nuclear' exists even if 0
        if 'nuclear' not in series.index:
            series.loc['nuclear'] = 0.0
    
        return series

    
    
    def objective_evaluation(self, n, nation, verbose=False):
        gen = n.generators[n.generators.index.str.startswith(nation)]
        su = n.storage_units[n.storage_units.index.str.startswith(nation)]
        sto = n.stores[n.stores.index.str.startswith(nation)]
        links = n.links[n.links.index.str.startswith(nation)]
        lines = n.lines[n.lines.bus0.str.startswith(nation) | n.lines.bus1.str.startswith(nation)]
    
        # Capital cost
        cap_gen = (gen.p_nom_opt * gen.capital_cost).sum()
        cap_su = (su.p_nom_opt * su.capital_cost).sum()
        cap_sto = (sto.e_nom_opt * sto.capital_cost).sum()
        cap_links = (links.p_nom_opt * links.capital_cost).sum()
        cap_lines = (lines.s_nom_opt * lines.capital_cost).sum()
    
        # Marginal costs
        mc_gen = (n.generators_t.p[gen.index] * gen.loc[gen.index, "marginal_cost"]).sum().sum()
        mc_su = (n.storage_units_t.p[su.index].abs() * su.loc[su.index, "marginal_cost"]).sum().sum()
        mc_sto = (n.stores_t.e[sto.index].abs() * sto.loc[sto.index, "marginal_cost"]).sum().sum()
        mc_links = (n.links_t.p0[links.index] * links.loc[links.index, "marginal_cost"]).sum().sum()
    
        total_capex = cap_gen + cap_su + cap_sto + cap_links + cap_lines
        total_opex = mc_gen + mc_su + mc_sto + mc_links
        total = total_capex + total_opex
    
        if verbose:
            print(f"\n--- Objective breakdown for {nation} ---")
            print(f"CAPEX:")
            print(f"  Generators:      {cap_gen:,.2f}")
            print(f"  Storage Units:   {cap_su:,.2f}")
            print(f"  Stores:          {cap_sto:,.2f}")
            print(f"  Links:           {cap_links:,.2f}")
            print(f"  Lines:           {cap_lines:,.2f}")
            print(f"Total CAPEX:       {total_capex:,.2f}")
            print(f"OPEX:")
            print(f"  Generators:      {mc_gen:,.2f}")
            print(f"  Storage Units:   {mc_su:,.2f}")
            print(f"  Stores:          {mc_sto:,.2f}")
            print(f"  Links:           {mc_links:,.2f}")
            print(f"Total OPEX:        {total_opex:,.2f}")
            print(f"Total Objective:   {total:,.2f}")
        
        return total / 1e9 # B€
    
    
    def marginal_prices_evaluation(self, n, nation, objective_data):
        """
        Calculate the average marginal price for buses in Italy, excluding buses containing 'h2' or 'battery'.
        The result is added to the objective_data dictionary.
        
        Parameters:
        - n: The network object
        - nation: The country code (e.g., 'IT' for Italy)
        - objective_data: A dictionary where the marginal price will be added
        """
        
        # Filter buses in Italy
        buses_in_italy = n.buses[n.buses.index.str.startswith(nation)].index
        
        # Extract the marginal prices for the buses in Italy (only those buses)
        marginal_prices = n.buses_t.marginal_price[buses_in_italy]
        
        # Filter out buses that contain 'h2' or 'battery'
        filtered_marginal_prices = marginal_prices.loc[:, ~marginal_prices.columns.str.contains('h2|battery', case=False, na=False)]
        
        # Calculate the average marginal price for the filtered buses
        average_marginal_price = filtered_marginal_prices.mean().mean()  # Mean across both rows and columns
        
        # Add the average marginal price to the objective_data dictionary
        objective_data = [objective_data, average_marginal_price]
        
        return objective_data




    def assert_objective(self, n, tol=1e-2):
        total = 0
        for nation in ['IT', 'FR', 'CH', 'AT', 'SI', 'GR', 'ME']:
            total += self.objective_evaluation(n, nation, verbose=True)
        
        expected = n.objective + n.objective_constant
        diff = total - expected
        print(f"\n=== Summary ===")
        print(f"Computed: {total:,.2f}")
        print(f"Expected: {expected:,.2f}")
        print(f"Difference: {diff:,.2f}")
    
        assert abs(diff) < tol, (
            f"Objective mismatch: computed {total:.2f}, expected {expected:.2f}, diff {diff:.2f}"
        )

         
    def plot_objective_with_avg_price(self, df, title, xlabel, ylabel_objective, ylabel_avg_price, output_name):
        """
        Plots a histogram for the 'objective' with stars indicating the average price for each scenario.
        It uses two y-axes, one for 'objective' and one for 'average_price'.
        
        Parameters:
        - df (DataFrame): The dataframe containing the objective and average_price.
        - title (str): The title of the plot.
        - xlabel (str): The label for the x-axis (scenarios).
        - ylabel_objective (str): The label for the objective y-axis.
        - ylabel_avg_price (str): The label for the average price y-axis.
        """
        fig, ax1 = plt.subplots(figsize=(12, 6))
    
        # Plot the histogram for the objective
        ax1.bar(df.columns, df.loc['objective'], color='skyblue', label='Objective', width=0.6, edgecolor='black')
        ax1.set_ylabel(ylabel_objective, fontsize=12, fontweight='bold')
        
        # Create the second y-axis for the average price
        ax2 = ax1.twinx()
        ax2.scatter(df.columns, df.loc['average_price'], color='orange', marker='*', s=300, label='Average Price')
        ax2.set_ylabel(ylabel_avg_price, fontsize=12, fontweight='bold')
    
        # Add title and customize the x-ticks
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_xticklabels([self.conversion_scenario.get(col, col) for col in df.columns], rotation=45, ha='right', fontsize=10)
        
        # Add legends
        ax1.legend(loc='upper left', title='Objective', fontsize=10)
        ax2.legend(loc='upper right', title='Average Price', fontsize=10)
    
        # Add gridlines
        ax1.grid(True, linestyle='--', alpha=0.7)
    
        # Adjust the layout and show the plot
        plt.tight_layout()
        plt.savefig(f"{self.output_folder}/{output_name}.png", format='png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_bar_chart(self, df, title, xlabel, ylabel, output_name):
        """
        Function to generate a vibrant bar chart with technologies on the x-axis and scenarios in the legend.
        
        Parameters:
        - df (DataFrame): Data to plot. Each column represents a technology and each row represents a scenario.
        - title (str): The title of the chart.
        - xlabel (str): The label for the x-axis.
        - ylabel (str): The label for the y-axis.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Rainbow color mapping for the scenarios
        rainbow_colors = [
            '#FF0000',  # Red
            '#FF7F00',  # Orange
            '#FFFF00',  # Yellow
            '#00FF00',  # Green
            '#0000FF',  # Blue
            '#4B0082',  # Indigo
            '#8B00FF'   # Violet
        ]
        
        # Create the bar chart with vibrant rainbow colors for the scenarios
        bars = df.plot(kind='bar', ax=ax, color=rainbow_colors[:len(df)], width=0.8, edgecolor='black')
        
        # Add title and axis labels
        ax.set_title(title, fontsize=14, fontweight='bold', color='#333333')
        ax.set_xlabel('')  # Remove x-label
        ax.set_ylabel(ylabel, fontsize=12, color='#333333')
        
        # Set the y-axis to a logarithmic scale
        # ax.set_yscale('log')
        
        # Add grid and adjust axis for better readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Rotate x-axis labels for readability and apply full technology names
        new_labels = [self.legend_map.get(label, label) for label in df.index]
        ax.set_xticks(np.arange(len(df.index)))  # Set the ticks to match the number of scenarios
        ax.set_xticklabels(new_labels, rotation=45, ha='right')  # Apply the full names for technologies
        
        # Add the legend with converted scenario names
        ax.legend([self.conversion_scenario.get(s, s) for s in df.columns], title="Scenarios", loc='best')
        
        # Add value labels on top of the bars (rounded without decimal)
        for bar in bars.patches:
            height = bar.get_height()
            # Format the height to display two decimals
            formatted_height = f'{height:.2f}'
            
            ax.annotate(formatted_height,  # Show the value with two decimal places
                        xy=(bar.get_x() + bar.get_width() / 2, height),  # Position of the label
                        xytext=(0, 5),  # Offset label position
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=4.5, color='black')
        
        # Improve layout
        plt.tight_layout()
        
        plt.savefig(f"{self.output_folder}/{output_name}.png", format='png', dpi=300, bbox_inches='tight')
        
        # Show the chart
        plt.show()
        
        
    def clean_hydrogen(self, p_nom_opt_data, p_dispatch_data):
        """
        Remove hydrogen data from p_nom_opt_data and p_dispatch_data, then plot a histogram
        with two y-axes, one for the size (p_nom_opt) and one for hydrogen production (p_dispatch).
        
        Parameters:
        - p_nom_opt_data: DataFrame containing the optimal power for different technologies
        - p_dispatch_data: DataFrame containing the dispatch data for different technologies
        """
        
        # Plotting the bar chart with two y-axes (one for p_nom_opt and one for p_dispatch)
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot p_nom_opt data (size of hydrogen installations) on the first axis
        ax1.bar(p_nom_opt_data.columns, p_nom_opt_data.loc['H2'], color='skyblue', label='Hydrogen Size', width=0.4, edgecolor='black', align='center')
        ax1.set_xlabel('Scenarios', fontsize=12)
        ax1.set_ylabel('Hydrogen Size (GW)', fontsize=12, color='black', fontweight='bold')
        ax1.set_yscale('log')  # Log scale for size (p_nom_opt)
        ax1.tick_params(axis='y', labelcolor='black')
        
        # Create the second y-axis for p_dispatch data (hydrogen production)
        ax2 = ax1.twinx()
        ax2.bar(p_dispatch_data.columns, p_dispatch_data.loc['H2'], color='orange', label='Hydrogen Production', width=0.4, edgecolor='black', align='edge')
        ax2.set_ylabel('Hydrogen Production (TWh)', fontsize=12, color='black', fontweight='bold')
        ax2.set_yscale('log')  # Log scale for production (p_dispatch)
        ax2.tick_params(axis='y', labelcolor='black')
        
        # Add title
        ax1.set_title('Hydrogen Size and Production for Different Scenarios', fontsize=14, fontweight='bold')
        
        # Add legends
        ax1.legend(loc='upper left', fontsize=10)
        ax2.legend(loc='upper right', fontsize=10)
        
        # Add gridlines
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Rotate x-axis labels for readability
        ax1.set_xticklabels([self.conversion_scenario.get(col, col) for col in p_nom_opt_data.columns], rotation=45, ha='right', fontsize=10)
        
        # Remove 'Hydrogen' row/column from both p_nom_opt_data and p_dispatch_data
        p_nom_opt_data_cleaned = p_nom_opt_data.drop('H2', axis=0)
        p_dispatch_data_cleaned = p_dispatch_data.drop('H2', axis=0)
        
        # Adjust the layout
        plt.tight_layout()
        plt.savefig(f"{self.output_folder}/Hydrogen.png", format='png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return p_nom_opt_data_cleaned, p_dispatch_data_cleaned

if __name__ == "__main__":
    name_list = ['2040_deit_biomass_limit', 'commodity_nze',
                 'low_electrical_demand', 'no_co2_emissions',
                 'nuclear_base', 'nuclear_no_emissions',
                 'no_biomass_limit']
    # name_list = ['nuclear_no_emissions']
    scenario_comparison = ScenarioComparison(name_list)
