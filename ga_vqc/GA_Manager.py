from . import simple_ga

# import .intermediate_ga
# import .advanced_ga
from . import GA_Support


def backend(config):
    if config["backend_type"] == "high":
        return simple_ga.Model(config)
    elif config["backend_type"] == "mid":
        # return .intermediate_ga.Model(config)
        pass
    elif config["backend_type"] == "low":
        # return .advanced_ga.Model(config)
        pass
    else:
        raise GA_Support.UnsupportedBackendDecorator(config["backend_type"])
