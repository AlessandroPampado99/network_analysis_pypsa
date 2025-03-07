# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 09:12:35 2025

@author: aless
"""

import os
import sys

# Determina il percorso assoluto della directory corrente
current_dir = os.path.dirname(os.path.abspath(__file__))

# Aggiungi il percorso relativo alla directory `../config`
config_path = os.path.abspath(os.path.join(current_dir, "../config"))

# Aggiungi il percorso a sys.path
sys.path.append(config_path)


from config import Config
from flowchart import Flowchart

config = Config()

flowchart = Flowchart(config)




    
