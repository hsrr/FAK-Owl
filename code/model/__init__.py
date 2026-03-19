from .agent import DeepSpeedAgent
from .openllama import OpenLLAMAPEFTModel
from .agent_multicls_only import DeepSpeedAgentMultiClsOnly
from .openllama_multicls_only import OpenLLAMAPEFTModelMultiClsOnly
from .openllama_multicls_aux_nogt import OpenLLAMAPEFTModelMultiClsAuxNoGT
# from .openllama_CLIP import OpenLLAMAPEFTModel_CLIP
from .ImageBind import models

def load_model(args):
    agent_name = args['models'][args['model']]['agent_name']
    model_name = args['models'][args['model']]['model_name']
    model = globals()[model_name](**args)
    agent = globals()[agent_name](model, args)
    return agent
