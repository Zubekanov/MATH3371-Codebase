import json
import os

class ConfigReader:    
    def __init__(self):
        pass
    
    @staticmethod
    def get_base_dir():
        self_file = os.path.abspath(__file__)
        src_dir = os.path.dirname(self_file)
        base_dir = os.path.dirname(src_dir)
        return base_dir
    
    @staticmethod
    def get_config_dir():
        base_dir = ConfigReader.get_base_dir()
        config_dir = os.path.join(base_dir, 'config')
        return config_dir
    
    @staticmethod
    def get_config_json(filename):
        if not filename.endswith('.json'):
            filename += '.json'
        config_dir = ConfigReader.get_config_dir()
        config_path = os.path.join(config_dir, filename)
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    
    @staticmethod
    def get_config_file(filename):
        if len(filename.split('.')) == 1:
            # Try to match filenames if no extension is provided.
            config_files = os.listdir(ConfigReader.get_config_dir())
            match_num = len([f for f in config_files if os.path.splitext(os.path.basename(f))[0] == filename])
            if match_num == 1:
                filename = [f for f in config_files if os.path.splitext(os.path.basename(f))[0] == filename][0]
            elif match_num == 0:
                raise FileNotFoundError(f'No file found matching {filename}.')
            else: # match_num > 1
                raise FileNotFoundError(f'Multiple files found matching {filename}, please specify the extension.')
            with open(os.path.join(ConfigReader.get_config_dir(), filename), 'r') as f:
                contents = f.read()
            return contents
        else:
            config_dir = ConfigReader.get_config_dir()
            config_path = os.path.join(config_dir, filename)
            with open(config_path, 'r') as f:
                contents = f.read()
            return contents
        