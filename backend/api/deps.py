from backend.services.alexnet_service import AlexNetService
from backend.services.backprop_service import BackpropService
from backend.services.deep_service import DeepService
from backend.services.gd_service import GdService
from backend.services.lms_service import LmsService
from backend.services.mlp_service import MlpService
from backend.services.perceptron_service import PerceptronService
from backend.services.scale_service import ScaleService
from backend.services.transformer_service import TransformerService

perceptron_service = PerceptronService()
lms_service = LmsService()
mlp_service = MlpService()
gd_service = GdService()
backprop_service = BackpropService()
deep_service = DeepService()
alexnet_service = AlexNetService()
transformer_service = TransformerService()
scale_service = ScaleService()
