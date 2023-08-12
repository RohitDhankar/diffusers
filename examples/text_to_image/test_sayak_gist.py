
# Source -- GIST from -- SAYAK --> https://gist.github.com/sayakpaul/a57a86ee7419ac3e7a7879fd100e8d06

"""
python test_sayak_gist.py --pipeline_id CompVis/stable-diffusion-v1-4

Examples:
    (1) python benchmark_distilled_sd.py --pipeline_id CompVis/stable-diffusion-v1-4
    (2) python benchmark_distilled_sd.py --pipeline_id CompVis/stable-diffusion-v1-4 --vae_path sayakpaul/taesd-diffusers
    (3) python benchmark_distilled_sd.py --pipeline_id nota-ai/bk-sdm-small
    (4) python benchmark_distilled_sd.py --pipeline_id nota-ai/bk-sdm-small --vae_path sayakpaul/taesd-diffusers 
"""

import argparse , time , torch
from diffusers import AutoencoderTiny, DiffusionPipeline

NUM_ITERS_TO_RUN = 3
NUM_INFERENCE_STEPS = 2#25
NUM_IMAGES_PER_PROMPT = 1 #4
#PROMPT = "a golden vase with different flowers"
PROMPT = "pencil sketch simple circle "
SEED = 0

#torch_device = "cuda" ## Original HugginFace Code 
torch_device1 = torch.device("cpu")
print("--torch_device1-",torch_device1)
torch_device2 = torch.device("cuda")
print("--torch_device2-",torch_device2)

"""
Pipelines loaded with `torch_dtype=torch.float16` cannot run with `cpu` device. It is not recommended to move them to 
`cpu` as running them will fail. Please make sure to use an accelerator to run the pipeline in inference, 
due to the lack of support for`float16` operations on this device in PyTorch. 
Please, remove the `torch_dtype=torch.float16` argument, or use another device for inference.

|   0  GeForce GTX 1650    On   | 00000000:01:00.0  On |                  N/A |
|  0%   42C    P8     4W /  75W |    314MiB /  3910MiB |      7%      Default |
|                               |                      |                  N/A 

"""

def load_pipeline(pipeline_id, vae_path=None):
    pipe = DiffusionPipeline.from_pretrained(pipeline_id, torch_dtype=torch.float16)
    pipe = pipe.to(torch_device2)#("cuda")

    if vae_path is not None:
        pipe.vae = AutoencoderTiny.from_pretrained(
            vae_path, torch_dtype=torch.float16
        ).to(torch_device2)#("cuda")

    return pipe

def run_inference(args):
    torch.cuda.reset_peak_memory_stats()
    pipe = load_pipeline(args.pipeline_id, args.vae_path)
    pipe.enable_attention_slicing() #GPU_MEMORY_OPTIMIZATION--->> https://huggingface.co/docs/diffusers/optimization/fp16

    start = time.time_ns()
    #for _ in range(NUM_ITERS_TO_RUN):
    for iter_k in range(NUM_ITERS_TO_RUN):
        print("--[INFO]---ITER_NUM-->",iter_k)
        images = pipe(
            PROMPT,
            num_inference_steps=NUM_INFERENCE_STEPS,
            generator=torch.manual_seed(SEED),
            num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
        ).images
        print("--[INFO]-type(pipe)--->",type(pipe))

    end = time.time_ns()
    mem_bytes = torch.cuda.max_memory_allocated()
    mem_MB = int(mem_bytes / (10**6))
    print("--[INFO]-mem_MB->",mem_MB)

    total_time = f"{(end - start) / 1e6:.1f}"
    results = {
        "pipeline_id": args.pipeline_id,
        "total_time (ms)": total_time,
        "memory (mb)": mem_MB,
    }
    print("--[INFO]-results--aa->",results)
    if args.vae_path is not None:
        results.update({"vae_path": args.vae_path})
    print("--[INFO]-results--bb->",results)
    return results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline_id",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        required=True,
    )
    parser.add_argument("--vae_path", type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    results = run_inference(args)
    print(results)
