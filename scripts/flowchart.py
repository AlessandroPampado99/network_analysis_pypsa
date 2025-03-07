# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 10:32:32 2025

@author: aless
"""
import os
import sys

# Determina il percorso assoluto della directory corrente
current_dir = os.path.dirname(os.path.abspath(__file__))

# Aggiungi il percorso relativo alla directory `../config`
config_path = os.path.abspath(os.path.join(current_dir, "../config"))

# Aggiungi il percorso a sys.path
sys.path.append(current_dir)

from analysis import PyPSANetworkAnalyzer
from output_writer import OutputWriter
from sliders_generation import SlidersGeneration

class Flowchart():
    
    def __init__(self, config):
        self.network_name = config.get('NETWORK', 'network')
        
        self.init(config)
        
    def init(self, config):
        if isinstance(self.network_name, str):
            if self.network_name == 'everything':
                network_names = list()
                for file in os.listdir("../networks"):
                    # Controlla se il file ha estensione .nc
                    if file.endswith(".nc"):
                        network_names.append(file)  # Salva il percorso completo
                for name in network_names:
                    self.network_name = name
                    network_analyzer = PyPSANetworkAnalyzer(name, config)
                    output_writer = OutputWriter(self.network_name, network_analyzer, config) 
                
                self.animate(config, network_names, output_writer)
            else:
                network_analyzer = PyPSANetworkAnalyzer(self.network_name, config)
                output_writer = OutputWriter(self.network_name, network_analyzer, config)
            
        elif isinstance(self.network_name, list):
            network_names = self.network_name
            for name in network_names:
                self.network_name = name
                network_analyzer = PyPSANetworkAnalyzer(name, config)
                output_writer = OutputWriter(self.network_name, network_analyzer, config) 
            
            self.animate(config, network_names, output_writer)
            
            
        
    def animate(self, config, network_names, output_writer):
        networks_to_animate = config.get('ANIMATION', 'networks_to_animate')
        if networks_to_animate == 'everything':
            networks_to_animate = network_names
        if networks_to_animate != str():
            graphs_list = config.get('ANIMATION', 'graphs_to_animate')
            for graph in graphs_list:
                image_paths = [f"../results/{network}/{graph}.png" for network in networks_to_animate]
                sliders_generation = SlidersGeneration(image_paths, networks_to_animate, graph, output_writer.output_folder)
                
                    
                
                
        