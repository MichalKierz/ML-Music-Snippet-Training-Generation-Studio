import torch
import torch.nn as nn

from src.models.kv_cache import KVCache
from src.models.rmsnorm import RMSNorm
from src.models.token_decoder_block import TokenDecoderBlock


class MetadataTokenTransformer(nn.Module):
    def __init__(
        self,
        token_vocab_size: int,
        artist_vocab_size: int,
        genre_vocab_size: int,
        mood_vocab_size: int,
        key_vocab_size: int,
        title_vocab_size: int,
        pad_token_id: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_kv_heads: int | None = None,
        n_layers: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
        max_token_length: int = 2048,
        metadata_prefix_tokens: int = 8,
        use_genre_metadata: bool = True,
        use_year_metadata: bool = True,
        use_mood_metadata: bool = False,
        use_key_metadata: bool = False,
        use_bpm_metadata: bool = False,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.token_vocab_size = int(token_vocab_size)
        self.pad_token_id = int(pad_token_id)
        self.d_model = int(d_model)
        self.max_token_length = int(max_token_length)
        self.n_heads = int(n_heads)
        self.n_kv_heads = int(n_kv_heads) if n_kv_heads is not None else int(n_heads)
        self.n_layers = int(n_layers)
        self.ff_mult = int(ff_mult)
        self.dropout = float(dropout)
        self.metadata_prefix_tokens = max(int(metadata_prefix_tokens), 8)
        self.use_genre_metadata = bool(use_genre_metadata)
        self.use_year_metadata = bool(use_year_metadata)
        self.use_mood_metadata = bool(use_mood_metadata)
        self.use_key_metadata = bool(use_key_metadata)
        self.use_bpm_metadata = bool(use_bpm_metadata)
        self.rope_base = float(rope_base)
        self.condition_token_count = 8

        self.token_embedding = nn.Embedding(self.token_vocab_size, self.d_model, padding_idx=self.pad_token_id)
        self.artist_embedding = nn.Embedding(int(artist_vocab_size), self.d_model)
        self.genre_embedding = nn.Embedding(int(genre_vocab_size), self.d_model)
        self.mood_embedding = nn.Embedding(int(mood_vocab_size), self.d_model)
        self.key_embedding = nn.Embedding(int(key_vocab_size), self.d_model)
        self.title_embedding = nn.Embedding(int(title_vocab_size), self.d_model, padding_idx=0)

        self.year_projection = nn.Sequential(
            nn.Linear(1, self.d_model, bias=False),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model, bias=False),
        )

        self.position_projection = nn.Sequential(
            nn.Linear(1, self.d_model, bias=False),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model, bias=False),
        )

        self.bpm_projection = nn.Sequential(
            nn.Linear(1, self.d_model, bias=False),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model, bias=False),
        )

        self.input_dropout = nn.Dropout(self.dropout)
        self.blocks = nn.ModuleList(
            [
                TokenDecoderBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    n_kv_heads=self.n_kv_heads,
                    ff_mult=self.ff_mult,
                    dropout=self.dropout,
                    rope_base=self.rope_base,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.final_norm = RMSNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.token_vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def init_kv_cache(self) -> KVCache:
        return KVCache.create(num_layers=self.n_layers)

    def _build_title_state(self, title_tokens: torch.Tensor, title_length: torch.Tensor) -> torch.Tensor:
        title_emb = self.title_embedding(title_tokens)
        mask = torch.arange(title_tokens.shape[1], device=title_tokens.device).unsqueeze(0) < title_length.unsqueeze(1)
        masked = title_emb * mask.unsqueeze(-1)
        denom = torch.clamp(title_length.unsqueeze(1), min=1)
        return masked.sum(dim=1) / denom

    def _build_condition_tokens(
        self,
        artist_id: torch.Tensor,
        genre_id: torch.Tensor,
        mood_id: torch.Tensor,
        key_id: torch.Tensor,
        title_tokens: torch.Tensor,
        title_length: torch.Tensor,
        year_value: torch.Tensor,
        position_value: torch.Tensor,
        bpm_value: torch.Tensor,
    ) -> torch.Tensor:
        artist_state = self.artist_embedding(artist_id)
        title_state = self._build_title_state(title_tokens, title_length)
        genre_state = self.genre_embedding(genre_id) if self.use_genre_metadata else torch.zeros_like(artist_state)
        year_state = self.year_projection(year_value.unsqueeze(1)) if self.use_year_metadata else torch.zeros_like(artist_state)
        position_state = self.position_projection(position_value.unsqueeze(1))
        mood_state = self.mood_embedding(mood_id) if self.use_mood_metadata else torch.zeros_like(artist_state)
        key_state = self.key_embedding(key_id) if self.use_key_metadata else torch.zeros_like(artist_state)
        bpm_state = self.bpm_projection(bpm_value.unsqueeze(1)) if self.use_bpm_metadata else torch.zeros_like(artist_state)

        return torch.stack(
            [
                artist_state,
                title_state,
                genre_state,
                year_state,
                position_state,
                mood_state,
                key_state,
                bpm_state,
            ],
            dim=1,
        )

    def _build_full_attention_mask(
        self,
        batch_size: int,
        token_attention_mask: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor | None:
        if token_attention_mask is None:
            return None

        prefix_mask = torch.ones((batch_size, self.condition_token_count), dtype=torch.bool, device=device)
        token_attention_mask = token_attention_mask.to(torch.bool)
        return torch.cat([prefix_mask, token_attention_mask], dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        artist_id: torch.Tensor,
        genre_id: torch.Tensor,
        mood_id: torch.Tensor,
        key_id: torch.Tensor,
        title_tokens: torch.Tensor,
        title_length: torch.Tensor,
        year_value: torch.Tensor,
        position_value: torch.Tensor,
        bpm_value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, token_length = input_ids.shape
        device = input_ids.device

        cond_tokens = self._build_condition_tokens(
            artist_id=artist_id,
            genre_id=genre_id,
            mood_id=mood_id,
            key_id=key_id,
            title_tokens=title_tokens,
            title_length=title_length,
            year_value=year_value,
            position_value=position_value,
            bpm_value=bpm_value,
        )

        token_states = self.token_embedding(input_ids)
        x = torch.cat([cond_tokens, token_states], dim=1)
        x = self.input_dropout(x)

        total_length = x.shape[1]
        position_ids = torch.arange(total_length, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        full_attention_mask = self._build_full_attention_mask(
            batch_size=batch_size,
            token_attention_mask=attention_mask,
            device=device,
        )

        for layer_index, block in enumerate(self.blocks):
            x = block(
                x,
                position_ids=position_ids,
                attention_mask=full_attention_mask,
                kv_cache=None,
                layer_index=layer_index,
                use_cache=False,
            )

        x = self.final_norm(x)
        logits = self.lm_head(x[:, self.condition_token_count :, :])
        return logits

    def prefill(
        self,
        prompt_token_ids: torch.Tensor,
        artist_id: torch.Tensor,
        genre_id: torch.Tensor,
        mood_id: torch.Tensor,
        key_id: torch.Tensor,
        title_tokens: torch.Tensor,
        title_length: torch.Tensor,
        year_value: torch.Tensor,
        position_value: torch.Tensor,
        bpm_value: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        batch_size, prompt_length = prompt_token_ids.shape
        device = prompt_token_ids.device

        cond_tokens = self._build_condition_tokens(
            artist_id=artist_id,
            genre_id=genre_id,
            mood_id=mood_id,
            key_id=key_id,
            title_tokens=title_tokens,
            title_length=title_length,
            year_value=year_value,
            position_value=position_value,
            bpm_value=bpm_value,
        )

        token_states = self.token_embedding(prompt_token_ids)
        x = torch.cat([cond_tokens, token_states], dim=1)
        x = self.input_dropout(x)

        total_length = x.shape[1]
        position_ids = torch.arange(total_length, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

        if kv_cache is None:
            kv_cache = self.init_kv_cache()

        for layer_index, block in enumerate(self.blocks):
            x = block(
                x,
                position_ids=position_ids,
                attention_mask=None,
                kv_cache=kv_cache,
                layer_index=layer_index,
                use_cache=True,
            )

        x = self.final_norm(x[:, -1:, :])
        logits = self.lm_head(x).squeeze(1)
        return logits, kv_cache

    def decode_step(
        self,
        next_input_ids: torch.Tensor,
        kv_cache: KVCache,
    ) -> tuple[torch.Tensor, KVCache]:
        batch_size, step_length = next_input_ids.shape
        device = next_input_ids.device

        if step_length != 1:
            raise ValueError("decode_step expects exactly one token per batch item.")

        x = self.token_embedding(next_input_ids)
        x = self.input_dropout(x)

        position_ids = torch.full(
            (batch_size, 1),
            fill_value=int(kv_cache.sequence_length),
            dtype=torch.long,
            device=device,
        )

        for layer_index, block in enumerate(self.blocks):
            x = block(
                x,
                position_ids=position_ids,
                attention_mask=None,
                kv_cache=kv_cache,
                layer_index=layer_index,
                use_cache=True,
            )

        x = self.final_norm(x)
        logits = self.lm_head(x).squeeze(1)
        return logits, kv_cache