from __future__ import annotations
from typing_extensions import override
import torch.nn.functional as F
from .. import Module, Embedding, Linear, RMSNorm
from ...model import Config
import torch
from ...util.tensor import get_for_device, to2
from ...tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX


class Gemma4VisionPatchEmbedder(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        hidden_size: int,
        patch_dim: int,
        position_embedding_size: int,
        out_dtype: torch.dtype = torch.float
    ):
        super().__init__(config, key, None)
        self.hidden_size = hidden_size
        self.position_embedding_size = position_embedding_size
        self.position_embedding_key = f"{key}.position_embedding_table"
        self.position_embedding_table = None
        self.position_embedding_numel = 0
        self.out_dtype = out_dtype

        self.input_proj = Linear(
            config = config,
            key = f"{key}.input_proj",
            in_features = patch_dim,
            out_features = hidden_size,
            qmap = None,
            out_dtype = torch.half,
            pad_to = 1,
        )
        self.register_submodule(self.input_proj)


    @override
    def optimizer_targets(self):
        return []


    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        self.position_embedding_table = self.config.stc.get_tensor(
            self.position_embedding_key,
            device,
            float2half = True,
            allow_bf16 = True,
        )
        self.position_embedding_numel = self.position_embedding_table.numel()


    @override
    def unload(self):
        super().unload()
        self.position_embedding_table = None
        self.position_embedding_numel = 0


    @override
    def weights_numel(self):
        return super().weights_numel() + self.position_embedding_numel


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:

        # Pixel values in range -1..1
        x = 2.0 * (x - 0.5)
        y = self.input_proj.forward(x.half(), params, out_dtype = torch.half)

        # Position IDs
        position_ids = get_for_device(params, "position_ids", self.device)
        pos_x = position_ids[..., 0].reshape(-1)
        pos_y = position_ids[..., 1].reshape(-1)

        # Table is 2x 1D learned embeddings (for x and y tile index, respectively)
        table = self.position_embedding_table
        pos_emb = table[0].index_select(0, pos_x) + table[1].index_select(0, pos_y)
        pos_emb = pos_emb.view(position_ids.shape[0], position_ids.shape[1], self.hidden_size)

        y = to2(y, out_dtype, self.out_dtype)
        y += pos_emb
        return y


class Gemma4VisionPooler(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        hidden_size: int,
        key_std_bias: str | None = None,
        key_std_scale: str | None = None,
    ):
        super().__init__(config, key, None)
        self.hidden_size = hidden_size
        self.std_bias_key = f"{key}.{key_std_bias}" if key_std_bias else None
        self.std_scale_key = f"{key}.{key_std_scale}" if key_std_scale else None
        self.std_bias = None
        self.std_scale = None
        self.numel = 0
        assert bool(self.std_bias_key) == bool(self.std_scale_key), \
            "Must have both std_bias and std_scale or neither"
        self.has_bias_scale = bool(self.std_bias_key)


    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        if self.has_bias_scale:
            self.std_bias = self.config.stc.get_tensor(self.std_bias_key, device, allow_bf16 = True)
            self.std_scale = self.config.stc.get_tensor(self.std_scale_key, device, allow_bf16 = True)


    @override
    def weights_numel(self):
        return 2 * self.hidden_size if self.has_bias_scale else 0


    @override
    def unload(self):
        super().unload()
        self.std_bias = None
        self.std_scale = None


    @override
    def optimizer_targets(self):
        return []


    @override
    def get_tensors(self):
        if self.has_bias_scale:
            return {
                self.std_bias_key: self.std_bias.contiguous(),
                self.std_scale_key: self.std_scale.contiguous(),
            }
        else:
            return {}


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        position_ids = get_for_device(params, "position_ids", self.device)
        output_length = int(params["image_output_length"])
        if output_length > x.shape[1]:
            raise ValueError(f"Cannot pool {x.shape[1]} patches to {output_length} soft tokens.")

        if x.shape[1] != output_length:
            input_seq_len = x.shape[1]
            k = int((input_seq_len // output_length) ** 0.5)
            k_squared = k ** 2
            if k_squared * output_length != input_seq_len:
                raise ValueError(f"Cannot pool {x.shape} to {output_length}: {k=}^2 mismatch")
            max_x = position_ids[..., 0].max(dim = -1, keepdim = True)[0] + 1
            kernel_idxs = torch.div(position_ids, k, rounding_mode = "floor")
            kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
            weights = F.one_hot(kernel_idxs.long(), output_length).float() / k_squared
            x = weights.transpose(1, 2) @ x

        x = x * (self.hidden_size ** 0.5)

        if self.has_bias_scale:
            x -= self.std_bias
            x *= self.std_scale

        return to2(x, out_dtype, torch.float)


class Gemma4TextInputEmbedding(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        vocab_size: int,
        hidden_size: int,
        pad_token_id: int,
        num_hidden_layers: int,
        hidden_size_per_layer_input: int,
        vocab_size_per_layer_input: int,
        out_dtype: torch.dtype = torch.float,
    ):
        super().__init__(config, key, None)
        self.module_name = "Gemma4TextInputEmbedding"
        self.pad_token_id = pad_token_id
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.out_dtype = out_dtype

        self.embedding = Embedding(
            config = config,
            key = f"{key}.embed_tokens",
            vocab_size = vocab_size,
            hidden_size = hidden_size,
            multiplier = hidden_size ** 0.5,
            out_dtype = out_dtype,
        )
        self.embedding_per_layer = Embedding(
            config = config,
            key = f"{key}.embed_tokens_per_layer",
            vocab_size = vocab_size_per_layer_input,
            hidden_size = num_hidden_layers * hidden_size_per_layer_input,
            multiplier = hidden_size_per_layer_input ** 0.5,
            out_dtype = out_dtype,
        )

        self.register_submodule(self.embedding)
        self.register_submodule(self.embedding_per_layer)

        # Keep the giant embedding tables on CPU, same as the standard embedding path.
        self.caps.update({
            "prefer_cpu": True,
        })


    @override
    def optimizer_targets(self):
        return []


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        input_ids = x
        mm_mask = input_ids >= FIRST_MM_EMBEDDING_INDEX
        if mm_mask.any():
            llm_input_ids = input_ids.clone()
            llm_input_ids[mm_mask] = self.pad_token_id
        else:
            llm_input_ids = input_ids

        text_embeds = self.embedding.forward(llm_input_ids, params, out_dtype = self.out_dtype)

        per_layer_tokens = self.embedding_per_layer.forward(
            llm_input_ids,
            params,
            out_dtype = self.out_dtype,
        )
        params["_gemma4_per_layer_token_inputs"] = per_layer_tokens.view(
            *llm_input_ids.shape,
            self.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

        if mm_mask.any():
            return self.embedding.forward(input_ids, params, out_dtype = out_dtype or self.out_dtype)

        return to2(text_embeds, out_dtype, self.out_dtype)


class Gemma4PerLayerInputProjector(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        hidden_size: int,
        num_hidden_layers: int,
        hidden_size_per_layer_input: int,
        rms_norm_eps: float,
        out_dtype: torch.dtype = torch.float,
        qmap: str | None = None,
    ):
        super().__init__(config, key, None)
        self.module_name = "Gemma4PerLayerInputProjector"
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.per_layer_model_projection_scale = hidden_size ** -0.5
        self.per_layer_input_scale = 2.0 ** -0.5
        self.out_dtype = out_dtype

        self.proj = Linear(
            config = config,
            key = f"{key}.per_layer_model_projection",
            in_features = hidden_size,
            out_features = num_hidden_layers * hidden_size_per_layer_input,
            qmap = qmap,
            out_dtype = out_dtype,
        )
        self.norm = RMSNorm(
            config = config,
            key = f"{key}.per_layer_projection_norm",
            rms_norm_eps = rms_norm_eps,
            out_dtype = out_dtype,
        )

        self.register_submodule(self.proj)
        self.register_submodule(self.norm)


    @override
    def optimizer_targets(self):
        return [self.proj.optimizer_targets()]


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        per_layer_tokens = params.pop("_gemma4_per_layer_token_inputs", None)
        if per_layer_tokens is not None:
            per_layer_tokens = per_layer_tokens.to(self.device, non_blocking = True)

        proj = self.proj.forward(x.half(), params, out_dtype = self.out_dtype)
        proj *= self.per_layer_model_projection_scale
        proj = proj.view(
            *x.shape[:-1],
            self.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        proj = self.norm.forward(proj, params, out_dtype = self.out_dtype)
        if per_layer_tokens is not None:
            per_layer_inputs = (proj + per_layer_tokens) * self.per_layer_input_scale
        else:
            per_layer_inputs = proj

        for idx in range(self.num_hidden_layers):
            params[f"_gemma4_per_layer_input.{idx}"] = per_layer_inputs[:, :, idx, :]

        return to2(x, out_dtype, self.out_dtype)


class Gemma4PerLayerInput(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        layer_idx: int,
        hidden_size: int,
        hidden_size_per_layer_input: int,
        rms_norm_eps: float,
        key_layer_scalar: str | None = None,
        out_dtype: torch.dtype = torch.float,
        gate_qmap: str | None = None,
        proj_qmap: str | None = None,
    ):
        super().__init__(config, key, None)
        self.module_name = "Gemma4PerLayerInput"
        self.layer_idx = layer_idx
        self.param_key = f"_gemma4_per_layer_input.{layer_idx}"
        self.key_layer_scalar = key_layer_scalar
        self.layer_scalar_t = None
        self.layer_scalar_f = None
        self.out_dtype = out_dtype

        self.gate = Linear(
            config = config,
            key = f"{key}.per_layer_input_gate",
            in_features = hidden_size,
            out_features = hidden_size_per_layer_input,
            qmap = gate_qmap,
            out_dtype = out_dtype,
        )
        self.proj = Linear(
            config = config,
            key = f"{key}.per_layer_projection",
            in_features = hidden_size_per_layer_input,
            out_features = hidden_size,
            qmap = proj_qmap,
            out_dtype = out_dtype,
        )
        self.norm = RMSNorm(
            config = config,
            key = f"{key}.post_per_layer_input_norm",
            rms_norm_eps = rms_norm_eps,
            out_dtype = out_dtype,
        )

        self.register_submodule(self.gate)
        self.register_submodule(self.proj)
        self.register_submodule(self.norm)


    @override
    def optimizer_targets(self):
        return [self.gate.optimizer_targets(), self.proj.optimizer_targets()]


    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        if self.key_layer_scalar:
            self.layer_scalar_t = self.config.stc.get_tensor(
                self.key + "." + self.key_layer_scalar,
                None,
                allow_bf16 = True,
                no_defer = True,
            )
            assert self.layer_scalar_t.numel() == 1
            self.layer_scalar_f = self.layer_scalar_t.float().item()


    @override
    def unload(self):
        super().unload()
        self.layer_scalar_t = None
        self.layer_scalar_f = None


    @override
    def get_tensors(self):
        tensors = {}
        if self.key_layer_scalar is not None:
            tensors[self.key + "." + self.key_layer_scalar] = self.layer_scalar_t.data.contiguous()
        return tensors


    @override
    def weights_numel(self):
        return super().weights_numel() + (1 if self.key_layer_scalar is not None else 0)


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        residual = x
        per_layer_input = params.get(self.param_key)
        if per_layer_input is not None:
            if per_layer_input.device != self.device:
                per_layer_input = per_layer_input.to(self.device, non_blocking = True)

            y = self.gate.forward(x.half(), params, out_dtype = self.out_dtype)
            y = F.gelu(y, approximate = "tanh")
            y *= per_layer_input
            y = self.proj.forward(y.half(), params, out_dtype = self.out_dtype)
            y = self.norm.forward(y, params, out_dtype = self.out_dtype)
            x = residual + y
        if self.layer_scalar_f is not None:
            x *= self.layer_scalar_f
        return to2(x, out_dtype, self.out_dtype)


class Gemma4DecoderLayer(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        layer_idx: int,
        hidden_size: int,
        attn_norm: Module,
        attn: Module,
        attn_post_norm: Module,
        mlp_norm: Module,
        mlp: Module,
        mlp_post_norm: Module,
        hidden_size_per_layer_input: int = 0,
        rms_norm_eps: float | None = None,
        key_layer_scalar: str | None = None,
        out_dtype: torch.dtype = torch.float,
        gate_qmap: str | None = None,
        proj_qmap: str | None = None,
    ):
        super().__init__(config, key, None)
        self.layer_idx = layer_idx
        self.attn_norm = attn_norm
        self.attn = attn
        self.attn_post_norm = attn_post_norm
        self.mlp_norm = mlp_norm
        self.mlp = mlp
        self.mlp_post_norm = mlp_post_norm
        self.key_layer_scalar = key_layer_scalar
        self.layer_scalar_t = None
        self.layer_scalar_f = None
        self.out_dtype = out_dtype

        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.param_key = f"_gemma4_per_layer_input.{layer_idx}" if hidden_size_per_layer_input else None

        self.ple_gate = None
        self.ple_proj = None
        self.ple_norm = None
        if hidden_size_per_layer_input:
            assert rms_norm_eps is not None, "Gemma4DecoderLayer requires rms_norm_eps when PLE is enabled"
            self.ple_gate = Linear(
                config = config,
                key = f"{key}.per_layer_input_gate",
                in_features = hidden_size,
                out_features = hidden_size_per_layer_input,
                qmap = gate_qmap,
                out_dtype = out_dtype,
            )
            self.ple_proj = Linear(
                config = config,
                key = f"{key}.per_layer_projection",
                in_features = hidden_size_per_layer_input,
                out_features = hidden_size,
                qmap = proj_qmap,
                out_dtype = out_dtype,
            )
            self.ple_norm = RMSNorm(
                config = config,
                key = f"{key}.post_per_layer_input_norm",
                rms_norm_eps = rms_norm_eps,
                out_dtype = out_dtype,
            )

        self.register_submodule(self.attn_norm)
        self.register_submodule(self.attn)
        self.register_submodule(self.attn_post_norm)
        self.register_submodule(self.mlp_norm)
        self.register_submodule(self.mlp)
        self.register_submodule(self.mlp_post_norm)
        self.register_submodule(self.ple_gate)
        self.register_submodule(self.ple_proj)
        self.register_submodule(self.ple_norm)

        self.num_slices = mlp.num_slices if mlp else 1


    @override
    def optimizer_targets(self):
        a = self.attn.optimizer_targets() if self.attn else []
        m = self.mlp.optimizer_targets() if self.mlp else []
        p = []
        if self.ple_gate is not None:
            p = [self.ple_gate.optimizer_targets(), self.ple_proj.optimizer_targets()]
        return [a, m, p]


    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        if self.key_layer_scalar:
            self.layer_scalar_t = self.config.stc.get_tensor(
                self.key + "." + self.key_layer_scalar,
                None,
                allow_bf16 = True,
                no_defer = True,
            )
            assert self.layer_scalar_t.numel() == 1
            self.layer_scalar_f = self.layer_scalar_t.float().item()


    @override
    def unload(self):
        super().unload()
        self.layer_scalar_t = None
        self.layer_scalar_f = None


    @override
    def get_tensors(self):
        tensors = {}
        if self.key_layer_scalar is not None:
            tensors[self.key + "." + self.key_layer_scalar] = self.layer_scalar_t.data.contiguous()
        return tensors


    @override
    def weights_numel(self):
        return super().weights_numel() + (1 if self.key_layer_scalar is not None else 0)


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if self.attn:
            y = self.attn_norm.forward(x, params, out_dtype = torch.half) if self.attn_norm else x.half()
            y = self.attn.forward(y, params)
            if self.attn_post_norm:
                y = self.attn_post_norm.forward(y, params)
            x += y

        if self.mlp:
            residual = x
            params["residual"] = x
            y = self.mlp_norm.forward(x, params, out_dtype = torch.half) if self.mlp_norm else x.half()
            y = self.mlp.forward(y, params)
            if self.mlp_post_norm:
                y = self.mlp_post_norm.forward(y, params)
            x = residual + y

        if self.ple_gate is not None:
            residual = x
            per_layer_input = params.get(self.param_key)
            if per_layer_input is not None:
                if per_layer_input.device != self.device:
                    per_layer_input = per_layer_input.to(self.device, non_blocking = True)
                y = self.ple_gate.forward(x.half(), params, out_dtype = self.out_dtype)
                y = F.gelu(y, approximate = "tanh")
                y *= per_layer_input
                y = self.ple_proj.forward(y.half(), params, out_dtype = self.out_dtype)
                y = self.ple_norm.forward(y, params, out_dtype = self.out_dtype)
                x = residual + y

        if self.layer_scalar_f is not None:
            x *= self.layer_scalar_f

        return to2(x, out_dtype, self.out_dtype)
