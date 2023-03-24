import json
import numpy as np
import torch
import triton_python_backend_utils as pb_utils

from torch import autocast
from torch.utils.dlpack import to_dlpack, from_dlpack
from diffusers import StableDiffusionPipeline


class TritonPythonModel:

    def initialize(self, args):
        self.output_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(json.loads(args["model_config"]),
                                               "generated_image")["data_type"])
        
        self.model_dir = args['model_repository']
    
        device='cuda'
        self.pipe = StableDiffusionPipeline.from_pretrained(f'{self.model_dir}/stable_diff',
                                                            torch_dtype=torch.float16).to(device)
        
        # This line of code does offload of model parameters to the CPU and only pulls them into the GPU as they are needed
        # Not tested with MME, since it will likely provoke CUDA OOM errors.
        #self.pipe.enable_sequential_cpu_offload()

    def execute(self, requests):
        responses = []
        for request in requests:
            inp = pb_utils.get_input_tensor_by_name(request, "prompt")
            input_text = inp.as_numpy()[0][0].decode()
            
            with torch.no_grad(), torch.autocast("cuda"):
                image_array = self.pipe(input_text,output_type='numpy').images
                
            decoded_image = (image_array * 255).round().astype("uint8")
            
            inference_response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor(
                    "generated_image",
                    decoded_image
                )
            ])
            
            responses.append(inference_response)
        
        return responses