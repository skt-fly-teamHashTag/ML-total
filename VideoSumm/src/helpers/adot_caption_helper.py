import numpy as np
import torch
import numpy as np
from fairseq import utils, tasks
from fairseq import checkpoint_utils
from OFA.utils.eval_utils import eval_caption
# from utils.eval_utils import eval_step
from OFA.tasks.mm_tasks.caption import CaptionTask
from OFA.models.ofa import OFAModel
from PIL import Image
from torchvision import transforms

import random
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode


def encode_text(text, length=None, append_bos=False, append_eos=False):
    s = task.tgt_dict.encode_line(
        line=task.bpe.encode(text),
        add_if_not_exist=False,
        append_eos=False
    ).long()
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s

# Construct input for caption task
def construct_sample(image: Image):
    patch_image = patch_resize_transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])
    src_text = encode_text(" what does the image describe?", append_bos=True, append_eos=True).unsqueeze(0)
    src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
    sample = {
        "id":np.array(['42']),
        "net_input": {
            "src_tokens": src_text,
            "src_lengths": src_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask
        }
    }
    return sample
  
# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t

def captioning(images):
    sentences = [] 

    # Register refcoco task
    tasks.register_task('caption', CaptionTask)

    # turn on cuda if GPU is available
    use_cuda = torch.cuda.is_available()
    # use fp16 only when GPU is available
    use_fp16 = False

    ## Build Model ##
    # Load pretrained ckpt & config
    overrides={"eval_cider":False, "beam":5, "max_len_b":16, "no_repeat_ngram_size":3, "seed":7}
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths('../models/ofa_pretrain/caption_base_best.pt'),
            arg_overrides=overrides
        )
    # Move models to GPU
    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Initialize generator
    generator = task.build_generator(models, cfg.generation)

    ## Image Preprocessing 
    # Image transform
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    patch_resize_transform = transforms.Compose(
        [
            lambda image: image.convert("RGB"),
            transforms.Resize((cfg.task.patch_image_size, cfg.task.patch_image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    # Text preprocess
    bos_item = torch.LongTensor([task.src_dict.bos()])
    eos_item = torch.LongTensor([task.src_dict.eos()])
    pad_idx = task.src_dict.pad()


    ## infer 
    for i in range(len(images)):
        img = images[i][0] # numpy array
        img = Image.fromarray(img) #numpy array to PIL image

        sample = construct_sample(img)
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample

        with torch.no_grad():
            result, scores = eval_caption(task, generator, models, sample)
        # print('Image{}, Caption: {}'.format(i, result[0]['caption']))
    
        # translator = googletrans.Translator()
        str1 = result[0]['caption']
        sentences.append(str1)

    return sentences 