from .simple_Model import Model as SimpleModel
# from .intermediate_Model import Model as IntermediateModel
# from .advanced_Model import Model as AdvancedModel

from . import GA_Support


def backend(config):
    if config.backend_type == "high":
        return SimpleModel(config)
    elif config.backend_type == "mid":
        # return IntermediateModel(config)
        pass
    elif config.backend_type == "low":
        # return AdvancedModel(config)
        pass
    else:
        raise GA_Support.UnsupportedBackendDecorator(config.backend_type)
