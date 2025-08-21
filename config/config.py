# config/config.py
import logging
from configparser import ConfigParser
from pathlib import Path
import ast

class Config:
    def __init__(self, filename: str = "application.ini"):
        self.log = logging.getLogger(__name__)
        if not self.log.handlers:
            logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

        self.parser = ConfigParser()
        self.config = {}

        # Read INI relative to this file, not to cwd
        self.config_path = Path(__file__).resolve().with_name(filename)
        loaded = self.parser.read(self.config_path, encoding="utf-8")
        if loaded:
            self.log.info("Loaded config: %s", loaded[0])
        else:
            self.log.error("Config file NOT found at: %s", self.config_path)

        self.init()

    # --- keep your logic, just safer parse_value ---
    def init(self):
        self.set_attributes()

    def set_attributes(self):
        for section in self.parser.sections():
            self.config[section] = {}
            for key in self.parser[section]:
                value = self.parse_value(self.parser[section][key])
                self.config[section][key] = value

    def parse_value(self, value):
        # Safer than eval
        try:
            evaluated_value = ast.literal_eval(value)
            if not isinstance(evaluated_value, str):
                return evaluated_value
        except Exception:
            pass
        return value

    def get(self, section, key, default=None):
        return self.config.get(section, {}).get(key, default)
