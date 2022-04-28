def read_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    from easydict import EasyDict as edict

    with open(filename, 'r') as f:
        cfg_dict = yaml.load(f, Loader=yaml.Loader)
        cfg_edict = edict(cfg_dict)
    return cfg_edict, cfg_dict

def save_to_file(path, document):

    import yaml   

    with open(path, 'w') as file:
        documents = yaml.dump(document, file, default_flow_style=False, sort_keys=False)

def save_file_to_file(output_path, input_path):
    
    import yaml   
    
    with open(input_path, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.Loader)

    with open(output_path, 'w') as file:
        documents = yaml.dump(yaml_cfg, file)
