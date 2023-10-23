import yaml


def read_params(config_path):
    """
    read parameters from params.yaml
    input: params.yaml
    output: parameters as dict
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config
