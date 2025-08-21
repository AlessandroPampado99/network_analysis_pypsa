# scripts/main.py
from .paths import ROOT, project_path
from config import Config            # thanks to config/__init__.py
from .flowchart import Flowchart

def main():
    # Small verbosity
    print("Project root:", ROOT)

    # Load config (reads config/application.ini)
    cfg = Config()

    # Example: build a robust path to the network file
    nc_path = project_path("networks", "Italy2019_uc_davide.nc")
    print("Using network file:", nc_path)

    # Do your work
    fc = Flowchart(cfg)
    # fc.run(nc_path)  # or whatever your API is

if __name__ == "__main__":
    main()
