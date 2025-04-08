# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 09:11:47 2025

@author: aless
"""

import logging
from configparser import ConfigParser

# Class for the initial configuration of the project
class Config():
    def __init__(self):
        self.parser = ConfigParser()
        self.parser.read("C:\\Users\\aless\\Desktop\\PhD_Pisa\\2025_01_03\\network_analysis\\config\\application.ini")
        
        self.config = {}  # Dizionario che conterrà la configurazione
        self.init()
        
#%% Section for the workflow of the system
    def init(self):
        self.set_attributes()

#%% Method to dinamically read the parameters in the config file
    def set_attributes(self):
        for section in self.parser.sections():
            self.config[section] = {}  # Crea un dizionario per ogni sezione
            for key in self.parser[section]:
                value = self.parse_value(self.parser[section][key])
                self.config[section][key] = value  # Aggiunge chiave e valore alla sezione

#%% Method to understand the type of the parsed value
    def parse_value(self, value):
        try:
            # Try to evaluate the value
            evaluated_value = eval(value)
            # If the evaluated value is not a string, return it
            if not isinstance(evaluated_value, str):
                return evaluated_value
        except:
            # If evaluation fails, return the original string
            pass
        return value

#%% Method to retrieve a value from the config dictionary
    def get(self, section, key, default=None):
        return self.config.get(section, {}).get(key, default)
