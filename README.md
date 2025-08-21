# network_analysis_pypsa

Self-contained project for PyPSA-based network analysis.  
Run from the project **root**; do not execute `scripts/main.py` directly.

---

## Quick start

```bash
git clone <REPO_URL>
cd network_analysis_pypsa

# (optional) create venv
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# (optional) install deps if requirements.txt exists
pip install -r requirements.txt

# run
python main.py
# or
python -m scripts.main

-----------------
Project structure
-----------------

network_analysis_pypsa/
├─ main.py                    # tiny runner that calls scripts.main
├─ scripts/
│  ├─ __init__.py
│  ├─ __main__.py            # enables: python -m scripts
│  ├─ paths.py               # ROOT + project_path helper
│  ├─ main.py                # real entrypoint (imports Config, Flowchart, ...)
│  ├─ flowchart.py
│  ├─ analysis.py
│  └─ ...                    # other modules
├─ config/
│  ├─ __init__.py
│  ├─ config.py              # Config class
│  └─ application.ini        # configuration file
├─ networks/
│  ├─ Italy2019_nouc_davide.nc
│  └─ Italy2019_uc_davide.nc
├─ output/
└─ .vscode/
   └─ launch.json            # (optional) VS Code debug config

---------------------
Paths & configuration
---------------------

All file paths are resolved relative to project root using scripts/paths.py:

from scripts.paths import project_path
nc = project_path("networks", "Italy2019_uc_davide.nc")


The Config class reads config/application.ini relative to config/ (not to the current working directory).

This design makes execution independent from the working directory (Spyder, VS Code, CLI, etc.).

----------
How to run
----------

Preferred: python main.py from the project root.

Alternative: python -m scripts.main from the project root.

Do not run scripts/main.py directly: package-relative imports would break.

-----------
Debugging
-----------

VS Code

Open the repo folder in VS Code.

Use the provided .vscode/launch.json or create one like this:

{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Run project main.py",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/main.py",
      "cwd": "${workspaceFolder}",
      "console": "integratedTerminal",
      "justMyCode": true
    },
    {
      "name": "Run module scripts.main",
      "type": "python",
      "request": "launch",
      "module": "scripts.main",
      "cwd": "${workspaceFolder}",
      "console": "integratedTerminal",
      "justMyCode": true
    }
  ]
}


Set breakpoints and press F5.

Spyder

Open the repo as a Project (File → Open project…).

Open main.py at the root and use Debug file.

Working directory can be either “file’s directory” or “current working directory”; paths are root-anchored, so both work.

------
Notes
------

No installation needed for local packages (scripts, config); they are part of the repo.

Avoid sys.path.append; packages + root anchoring is more robust.

If you ever want to reuse this project from outside, add a pyproject.toml and run pip install -e . (not required here).

---------------
Troubleshooting
---------------

FileNotFoundError for .nc files
Check the file exists under networks/ and that you’re running from the repo root (python main.py).

ImportError for config/scripts
Ensure __init__.py files exist and you’re not executing scripts/main.py directly.

Config not loading
Make sure config/application.ini exists and has proper [sections].