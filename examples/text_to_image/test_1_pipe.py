
from diffusers import DDPMPipeline , UNet2DModel , DDPMScheduler
import sys , torch
import numpy as np
from PIL import Image

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import UniPCMultistepScheduler
from tqdm.auto import tqdm


# ddpm = DDPMPipeline.from_pretrained("google/ddpm-cat-256").to("cuda")
# pil_image = ddpm(num_inference_steps=25).images[0]
# print("  "*90)
# print("---type(image)")
# print(type(pil_image))
# pil_image.show()
# pil_image.save("cat_1_1.png")

# num_img = np.asarray(pil_image)
# print("  "*90)
# print("---type(num_img)")
# print(type(num_img))


#torch_device = "cuda" ## Original HugginFace Code 
torch_device1 = torch.device("cpu")
print("--torch_device1-",torch_device1)
torch_device2 = torch.device("cuda")
print("--torch_device2-",torch_device2)


def denoise_func_1():
    """
    """

    scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
    model = UNet2DModel.from_pretrained("google/ddpm-cat-256").to("cuda")
    scheduler.set_timesteps(150)
    print("----time-steps---",scheduler.timesteps)
    #
    sample_size = model.config.sample_size
    noise = torch.randn((1, 3, sample_size, sample_size)).to("cuda")
    print("--type(noise---",type(noise))
    #
    input = noise

    for t_step in scheduler.timesteps:
        with torch.no_grad():
            noisy_residual = model(input, t_step).sample
            print("--type(noisy_residual--",type(noisy_residual))
        previous_noisy_sample = scheduler.step(noisy_residual, t_step, input).prev_sample
        print("--type(previous_noisy_sample-",type(previous_noisy_sample))
        input = previous_noisy_sample

        pil_image1 = (input / 2 + 0.5).clamp(0, 1)
        pil_image1 = pil_image1.cpu().permute(0, 2, 3, 1).numpy()[0]
        pil_image1 = Image.fromarray((pil_image1 * 255).round().astype("uint8"))
        print("  "*90)
        print("---type(image)")
        print(type(pil_image1))
        pil_image1.show()
        pil_image1.save("cat_2_2.png")


def stable_diffusion_test_pipe():
    """
    """
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    print("--type(vae--->",type(vae))

    tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
    print("--type(tokenizer--->",type(tokenizer))
    ##<class 'transformers.models.clip.tokenization_clip.CLIPTokenizer'>
    ##https://github.com/RohitDhankar/transformers/blob/a6e6b1c622d8d08e2510a82cb6266d7b654f1cbf/src/transformers/models/clip/tokenization_clip.py#L272


    text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder")
    print("--type(text_encoder--->",type(text_encoder))
    ##<class 'transformers.models.clip.modeling_clip.CLIPTextModel'>


    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
    print("--type(unet--->",type(unet))

    scheduler = UniPCMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    print("--type(scheduler--->",type(scheduler))
    # https://github.com/Sygil-Dev/sygil-webui

    #torch_device = "cuda" ## Original HugginFace Code 
    torch_device1 = torch.device("cpu")
    print("--torch_device1-",torch_device1)
    torch_device2 = torch.device("cuda")
    print("--torch_device2-",torch_device2)

    # #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # vae.to(torch_device)
    # text_encoder.to(torch_device)
    # unet.to(torch_device)
    # #
    prompt = ["god stranded on a tropical island"]
    height = 512  # default height of Stable Diffusion
    width = 512  # default width of Stable Diffusion
    num_inference_steps = 250  # Number of denoising steps
    guidance_scale = 7.5  # Scale for classifier-free guidance
    generator = torch.manual_seed(0)  # Seed generator to create the inital latent noise
    batch_size = len(prompt)
    # #
    text_input = tokenizer(
    prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    print("--text_input--->",text_input)
    print("--text_input--->",text_input.input_ids)
    print("--text_input--text_input.attention_mask-->",text_input.attention_mask)
    ## input_ids + attention_mask

    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device1))[0]
        print("--type(text_embeddings-->",type(text_embeddings))
        print("--text_embeddings-->",text_embeddings)

    #
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device1))[0]
    print("--type(uncond_embeddings-->",type(uncond_embeddings))
    print("--uncond_embeddings-->",uncond_embeddings)
    #
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    #
    latents = torch.randn((batch_size, unet.in_channels, height // 8, width // 8),generator=generator,)
    print("--type(latents-1---->",type(latents))
    print("--latents-1---->",latents)
    
    # latents = latents.to(torch_device)
    # #
    # latents = latents * scheduler.init_noise_sigma
    # print("--type(latents-2->",latents)
    # #
    # scheduler.set_timesteps(num_inference_steps)
    # for t_step in tqdm(scheduler.timesteps):
    #     # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    #     latent_model_input = torch.cat([latents] * 2)

    #     latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t_step)

    #     # predict the noise residual
    #     with torch.no_grad():
    #         noise_pred = unet(latent_model_input, t_step, encoder_hidden_states=text_embeddings).sample

    #     # perform guidance
    #     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    #     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    #     # compute the previous noisy sample x_t -> x_t-1
    #     latents = scheduler.step(noise_pred, t_step, latents).prev_sample
    #     print("--type(latents-3->",latents)

    #     # scale and decode the image latents with vae
    #     latents = 1 / 0.18215 * latents
    #     with torch.no_grad():
    #         image = vae.decode(latents).sample

    #     image = (image / 2 + 0.5).clamp(0, 1)
    #     image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    #     images = (image * 255).round().astype("uint8")
    #     pil_images = [Image.fromarray(image) for image in images]
    #     pil_image1 = pil_images[0]
    #     print("  "*90)
    #     print("---type(stable_diff_RES_IMAGE--->")
    #     print(type(pil_image1))
    #     pil_image1.show()
    #     pil_image1.save("res_stable_diff_.png")


if __name__ == '__main__':
    #denoise_func_1()
    stable_diffusion_test_pipe()






# img = Image.open(sys.argv[1]).convert('L')

# im = numpy.array(img)
# fft_mag = numpy.abs(numpy.fft.fftshift(numpy.fft.fft2(im)))

# visual = numpy.log(fft_mag)
# visual = (visual - visual.min()) / (visual.max() - visual.min())

# result = Image.fromarray((visual * 255).astype(numpy.uint8))
# result.save('out.bmp')