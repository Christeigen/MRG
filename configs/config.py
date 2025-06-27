from dataclasses import dataclass, field

class DataConfig():
    def __init__(
        self,
        annotation_path = "data/annotation.json",
        threshold = 2,
        max_seq_length = 100,
        dataset_name = "updated_iu_xray",
        image_dir = "data/updated_iu_xray"
    ):
        super().__init__()
        self.annotation_path = annotation_path
        self.threshold = threshold
        self.max_seq_length = max_seq_length
        self.dataset_name = dataset_name
        self.image_dir = image_dir

class EncoderConfig():

    def __init__(
        self,
        rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
        enable_lora=True,
        clip_model_name="ViT-B/32",
        image_size=224,
        patch_size=16,
        **kwargs
    ):
        super().__init__()

        self.rank= rank
        self.lora_alpha = lora_alpha
        self.lora_dropout=lora_dropout
        self.enable_lora=enable_lora
        self.clip_model_name=clip_model_name
        self.image_size=image_size
        self.patch_size=patch_size

class DecoderConfig():

    def __init__(
        self,
        vocab_size=979,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=256,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

class PaliGemmaConfig():

    def __init__(
        self,
        text_config=None,
        ignore_index=-100,
        image_token_index=978,
        vocab_size=979,
        projection_dim=768,
        hidden_size=768,
        pad_token_id=0,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = EncoderConfig()
        self.text_config = text_config

        self.text_config = DecoderConfig(pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim