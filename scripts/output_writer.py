# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 15:11:32 2025

@author: aless
"""

from analysis import PyPSANetworkAnalyzer
import pandas as pd
import os
import openpyxl
import os
import sys

# Determina il percorso assoluto della directory corrente
current_dir = os.path.dirname(os.path.abspath(__file__))

# Aggiungi il percorso relativo alla directory `../config`
config_path = os.path.abspath(os.path.join(current_dir, "../config"))

# Aggiungi il percorso a sys.path
sys.path.append(config_path)


from config import Config

class OutputWriter():
    
    def __init__(self, network_name, network_analyzer, config):
        self.network_name = network_name
        self.network_analyzer = network_analyzer
        self.config = config
        
        if self.config.get("NETWORK", "comparison_output"):
            name = self.config.get("NETWORK", "comparison_name")
            self.output_folder = f"../results/comparison_{name}"
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)
        else:
            self.output_folder = f"../results/{network_name}"
        
        self.init()
        
    def init(self):
        self.export_obj_stat()
        
        self.export_size_increase()
        
        self.export_line_loading()
        self.export_line_expansion()
        
        self.export_load()
        
        
        
    def export_obj_stat(self):
        """
        Exports network statistics and optimization objectives to separate Excel files.
        - `objective` is appended as a new row if the file exists and the sheet has rows; otherwise, it writes a new sheet.
        - `statistics` is written to a new sheet named `statistics_<network_name>` in a file.
    
        Returns
        -------
        None
        """
        n = self.network_analyzer.network
        statistics = self.network_analyzer.statistics
        
        network_name = self.network_name
    
        # Prepare the objective DataFrame
        objective_operation = n.objective
        objective_design = n.objective_constant
        objective_df = pd.DataFrame(
            [[objective_design, objective_operation, objective_design + objective_operation]],
            columns=['design cost', 'operation cost', 'total cost'],
            index=[network_name]
        )
        
        self.export_objective(objective_df, 'objective')
        self.export_statistics(statistics)
        
    
    def export_size_increase(self):
        size_increase = self.network_analyzer.results['generators_grouped']['increase_by_carrier']
        size_increase.name = self.network_name
        size_increase = pd.DataFrame(size_increase.T)
        
        self.export_objective(pd.DataFrame(size_increase.T), "increase_by_carrier")
        
    
    def export_line_loading(self):
        for attr in ['max', 'mean']:
            line_loading = self.network_analyzer.results[f"line_loading_{attr}"]
            line_loading.index = line_loading.index.astype(int)
            line_loading = line_loading.sort_index()
            line_loading.name = self.network_name
            line_loading = pd.DataFrame(line_loading).T
            
            self.export_objective(line_loading, f"line_loading_{attr}")
        
    def export_line_expansion(self):
        line_expansion = self.network_analyzer.results['line_expansion']['line_expansion_relative']
        line_expansion.index = line_expansion.index.astype(int)
        line_expansion = line_expansion.sort_index()
        line_expansion.name = self.network_name
        line_expansion = pd.DataFrame(line_expansion).T
        
        
        self.export_objective(line_expansion, 'line_expansion')
    
    
    def export_load(self):
        total_load = self.network_analyzer.results['total_load']
        total_load.name = self.network_name
        total_load = pd.DataFrame(total_load).T
        
        self.export_objective(total_load, 'total_load')
        
    
    def export_objective(self, objective_df, output_name):
        """
        Export objective data to an Excel file.
        If the sheet 'objective' exists:
            - Check if the index of objective_df exists in the sheet.
            - Overwrite the row if the index exists.
            - Append a new row if the index does not exist.
        If the file or sheet does not exist, create them.
    
        Parameters
        ----------
        objective_df : pd.DataFrame
            DataFrame containing the objective data to export.
        """
        file_objective = f"{self.output_folder}/{output_name}.xlsx"
    
        if os.path.exists(file_objective):
            # Load existing data
            existing_data = pd.read_excel(file_objective, sheet_name=output_name, index_col=0)
    
            # Check if index already exists
            for index in objective_df.index:
                if index in existing_data.index:
                    # Replace the existing row
                    existing_data.loc[index] = objective_df.loc[index]
                else:
                    # Append the new row
                    existing_data = pd.concat([existing_data, objective_df.loc[[index]]])
    
            # Save the updated data
            with pd.ExcelWriter(file_objective, engine="openpyxl") as writer:
                existing_data.to_excel(writer, sheet_name=output_name)
        else:
            # If the file doesn't exist, create it and write the data
            with pd.ExcelWriter(file_objective, engine="openpyxl") as writer:
                objective_df.to_excel(writer, sheet_name=output_name)

    
    
    def export_statistics(self, statistics):
        file_statistics = f"{self.output_folder}/statistics.xlsx"
        # --- Handle statistics ---
        sheet_name = self.network_name.split('.')[0].split('s_')[-1]
    
        if os.path.exists(file_statistics):
            with pd.ExcelWriter(file_statistics, mode="a", engine="openpyxl", if_sheet_exists="overlay") as writer:
                statistics.to_excel(writer, sheet_name=sheet_name, index=True)
        else:
            with pd.ExcelWriter(file_statistics, engine="openpyxl") as writer:
                statistics.to_excel(writer, sheet_name=sheet_name, index=True)


#%% Use while __main__

if __name__ == '__main__':
    config = Config()
    network_name = 'base_s_39_elec_lvopt_1h_noext_2030.nc'
    config.config['NETWORK']['comparison_output'] = False
    network_analyzer =  PyPSANetworkAnalyzer(network_name, config)
    
    output_writer = OutputWriter(network_name,network_analyzer, config)
        