from typing import Mapping, Any
import copy
from collections import OrderedDict

import einops
import torch
import torch as th
import torch.nn as nn

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
    snr_embedding,
)
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock, UNetModel
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from utils.common import frozen_module
from .spaced_sampler import SpacedSampler
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from torchvision import transforms


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        use_snr=True,  # whether to use SNR information or not
        min_snr=0,     # provide the lowest possible SNR
        snr_scale=1,   # scale factor for the SNR value
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        self.use_snr = use_snr
        self.min_snr = min_snr
        self.snr_scale = snr_scale        

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        
        self.snr_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels + hint_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, snr, **kwargs):

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        
        # If we choose to use the SNR information
        if self.use_snr:
            snr_emb = snr_embedding(snr[:, 0]*self.snr_scale, self.model_channels, min_SNR=self.min_snr, repeat_only=False)
            snr_emb = self.snr_embed(snr_emb)
            emb = emb + snr_emb

        x = torch.cat((x, hint), dim=1)
        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlLDM(LatentDiffusion):

    def __init__(
        self,
        control_stage_config: Mapping[str, Any],
        control_key: str,
        sd_locked: bool,
        only_mid_control: bool,
        learning_rate: float,
        preprocess_config,
        use_lang,
        use_true_lang,
        use_replace,
        *args,
        **kwargs
    ) -> "ControlLDM":
        super().__init__(*args, **kwargs)
        # instantiate control module

        self.control_model: ControlNet = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.sd_locked = sd_locked
        self.only_mid_control = only_mid_control
        self.learning_rate = learning_rate
        self.control_scales = [1.0] * 13
        self.use_lang = use_lang
        self.use_true_lang = use_true_lang
        self.use_replace = use_replace

        # instantiate JSCC model
        self.preprocess_model = instantiate_from_config(preprocess_config)
        frozen_module(self.preprocess_model)

        # instantiate condition encoder, since our condition encoder has the same
        # structure with AE encoder, we just make a copy of AE encoder. please
        # note that AE encoder's parameters has not been initialized here.
        self.cond_encoder = nn.Sequential(OrderedDict([
            ("encoder", copy.deepcopy(self.first_stage_model.encoder)),  # cond_encoder.encoder
            ("quant_conv", copy.deepcopy(self.first_stage_model.quant_conv))  # cond_encoder.quant_conv
        ]))
        frozen_module(self.cond_encoder)

        if self.use_lang:
            self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, max_length=32
            )
            frozen_module(self.blip_model)

        self.transform_to_pil = transforms.ToPILImage()

    def apply_condition_encoder(self, control):
        c_latent_meanvar = self.cond_encoder(control * 2 - 1)
        c_latent = DiagonalGaussianDistribution(c_latent_meanvar).mode()  # only use mode
        c_latent = c_latent * self.scale_factor
        return c_latent

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        
        #x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        
        # Pass the input through the deep JSCC encoder and decoder
        # control = batch['jpg']
        # if bs is not None:
        #     control = control[:bs]
        # control = control.to(self.device)
        # #control = einops.rearrange(control, 'b h w c -> b c h w')
        # #control = ((control + 1) / 2).clamp_(0, 1)
        # control = control.to(memory_format=torch.contiguous_format).float()
        # lq = control
        
        # apply JSCC model and get the initial reconstruction
        # control_, SNR, _ = self.preprocess_model(control)
        
        # If we would like to extract text features from the initial reconstruction
        
        img_t = batch['jpg']
        if bs is not None:
            img_t = img_t[:bs]
        img_t = img_t.to(self.device)
        img_t = img_t.to(memory_format=torch.contiguous_format).float()

        img_init, SNR, _ = self.preprocess_model(img_t)

        if self.use_lang:
            input_img = []
            for b in range(img_t.shape[0]):
                if self.use_true_lang:
                    input_img.append(self.transform_to_pil(img_t[b]))
                else:
                    input_img.append(self.transform_to_pil(img_init[b]))
                
            inputs = self.processor(images=input_img, return_tensors="pt", max_length=32).to(self.device, torch.float16)
            generated_ids = self.blip_model.generate(**inputs)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

            if not self.use_true_lang and self.use_replace:
                for b in range(len(generated_text)):
                    generated_text[b] = (generated_text[b].replace("blurry", "high quality")
                                 .replace("a painting of", "a high quality image of")
                                 .replace("a photo of", "a high quality image of")
                                 .replace("a picture of", "a high quality image of"))
                    if "a high quality image of" not in generated_text[b]:
                        generated_text[b] = "a high quality image of " + generated_text[b]  
                                                 
            c = self.get_learned_conditioning(generated_text)            
        else:
            generated_text = batch['txt']
            c = self.get_learned_conditioning(generated_text) 

        # apply condition encoder
        c_latent = self.apply_condition_encoder(img_init)

        # Get the latents of the original image
        encoder_posterior = self.encode_first_stage(img_t*2-1)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        return z, dict(c_textual=[c], c_spatial=[c_latent], c_snr=[SNR], img_t=[img_t], img_init=[img_init], text=[generated_text])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        
        assert isinstance(cond, dict)

        diffusion_model = self.model.diffusion_model
        
        cond_txt = torch.cat(cond['c_textual'], 1)
        
        if cond['c_spatial'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(
                x=x_noisy, hint=torch.cat(cond['c_spatial'], 1),
                timesteps=t, context=cond_txt, snr=torch.cat(cond['c_snr'], 1),
            )
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, sample_steps=50):
        log = dict()
        z, c = self.get_input(batch, self.first_stage_key)
        log["img_init"] = c["img_init"][0]
        log["img_init_decoded"] = (self.decode_first_stage(c["c_spatial"][0]) + 1) / 2
        log["img_t"] = c["img_t"][0]
        log["img_t_decoded"] = (self.decode_first_stage(z) + 1) / 2
        log["text"] = (log_txt_as_img((512, 512), c["text"][0], size=16) + 1) / 2
        log["samples"] = self.sample_log(
            cond_img=c["img_init"][0],
            cond_snr=c["c_snr"][0],
            cond_text=c["c_textual"][0],
            steps=sample_steps
        )
        return log

    @torch.no_grad()
    def sample_log(self, cond_img, cond_snr, cond_text, steps):
        sampler = SpacedSampler(self)
        b, c, h, w = cond_img.shape
        shape = (b, self.channels, h // 8, w // 8)
        samples = sampler.sample(
            steps, shape, cond_img, cond_snr, cond_text=cond_text, positive_prompt="", negative_prompt="",
            cfg_scale=1.0, color_fix_type="wavelet"
        )
        return samples

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def validation_step(self, batch, batch_idx):
        # TODO:
        pass
    
    def on_save_checkpoint(self, checkpoint):
        # This method is called when saving a checkpoint
        # Modify checkpoint to contain only the parts that you want to save
        
        # Example: to only save the weights of "layer1"
        selective_state_dict = {k: v for k, v in self.state_dict().items() if 'blip_model' not in k}
        checkpoint['state_dict'] = selective_state_dict

        return checkpoint


