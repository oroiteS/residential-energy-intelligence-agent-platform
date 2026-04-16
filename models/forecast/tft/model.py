"""Baseline-aware TFT v2。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GatedLinearUnit(nn.Module):
    """TFT 中的门控线性单元。"""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value, gate = self.linear(inputs).chunk(2, dim=-1)
        return value * torch.sigmoid(gate)


class GateAddNorm(nn.Module):
    """门控残差与层归一化。"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        residual_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.glu = GatedLinearUnit(input_dim, output_dim)
        self.residual_projection = (
            nn.Identity()
            if residual_dim is None or residual_dim == output_dim
            else nn.Linear(residual_dim, output_dim)
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        return self.norm(self.glu(inputs) + self.residual_projection(residual))


class GatedResidualNetwork(nn.Module):
    """TFT 的 GRN 模块，当前不依赖静态上下文。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        resolved_output_dim = output_dim or input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, resolved_output_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = GateAddNorm(
            input_dim=resolved_output_dim,
            output_dim=resolved_output_dim,
            residual_dim=input_dim,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.elu(self.fc1(inputs))
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)
        return self.gate(hidden, inputs)


class VariableSelectionNetwork(nn.Module):
    """对连续变量执行逐时刻变量选择。"""

    def __init__(
        self,
        num_variables: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_variables = num_variables
        self.variable_projections = nn.ModuleList(
            [nn.Linear(1, hidden_size) for _ in range(num_variables)]
        )
        self.variable_grns = nn.ModuleList(
            [
                GatedResidualNetwork(
                    input_dim=hidden_size,
                    hidden_dim=hidden_size,
                    output_dim=hidden_size,
                    dropout=dropout,
                )
                for _ in range(num_variables)
            ]
        )
        self.weight_grn = GatedResidualNetwork(
            input_dim=num_variables,
            hidden_dim=hidden_size,
            output_dim=num_variables,
            dropout=dropout,
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sparse_weights = torch.softmax(self.weight_grn(inputs), dim=-1)
        transformed_variables: list[torch.Tensor] = []
        for variable_index in range(self.num_variables):
            single_variable = inputs[..., variable_index : variable_index + 1]
            projected = self.variable_projections[variable_index](single_variable)
            transformed = self.variable_grns[variable_index](projected)
            transformed_variables.append(transformed)

        stacked = torch.stack(transformed_variables, dim=-2)
        combined = torch.sum(sparse_weights.unsqueeze(-1) * stacked, dim=-2)
        return combined, sparse_weights


class ExpertResidualHead(nn.Module):
    """单个 expert residual 头。"""

    def __init__(self, input_dim: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.grn = GatedResidualNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_size,
            output_dim=hidden_size,
            dropout=dropout,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.grn(inputs)).squeeze(-1)


class TemporalFusionTransformer(nn.Module):
    """带 baseline gate 与 profile expert 的 TFT v2 主体。"""

    def __init__(
        self,
        encoder_input_dim: int,
        decoder_input_dim: int,
        hidden_size: int,
        lstm_layers: int,
        attention_heads: int,
        dropout: float,
        prediction_length: int,
        profile_dim: int,
        baseline_stat_dim: int,
        router_prior_weight: float = 1.0,
        global_gate_bias_init: float = -1.0,
        local_gate_bias_init: float = -1.0,
        return_attention_weights: bool = False,
    ) -> None:
        super().__init__()
        self.prediction_length = prediction_length
        self.profile_dim = profile_dim
        self.return_attention_weights = return_attention_weights
        self.router_prior_weight = router_prior_weight

        self.encoder_vsn = VariableSelectionNetwork(
            num_variables=encoder_input_dim,
            hidden_size=hidden_size,
            dropout=dropout,
        )
        self.decoder_vsn = VariableSelectionNetwork(
            num_variables=decoder_input_dim,
            hidden_size=hidden_size,
            dropout=dropout,
        )
        lstm_dropout = dropout if lstm_layers > 1 else 0.0
        self.encoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.post_lstm_gate = GateAddNorm(
            input_dim=hidden_size,
            output_dim=hidden_size,
            residual_dim=hidden_size,
        )
        self.enrichment_grn = GatedResidualNetwork(
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            output_dim=hidden_size,
            dropout=dropout,
        )
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.post_attention_gate = GateAddNorm(
            input_dim=hidden_size,
            output_dim=hidden_size,
            residual_dim=hidden_size,
        )
        self.positionwise_grn = GatedResidualNetwork(
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            output_dim=hidden_size,
            dropout=dropout,
        )
        self.pre_output_gate = GateAddNorm(
            input_dim=hidden_size,
            output_dim=hidden_size,
            residual_dim=hidden_size,
        )

        context_input_dim = hidden_size * 2 + profile_dim + baseline_stat_dim
        self.context_grn = GatedResidualNetwork(
            input_dim=context_input_dim,
            hidden_dim=hidden_size,
            output_dim=hidden_size,
            dropout=dropout,
        )
        self.router_grn = GatedResidualNetwork(
            input_dim=hidden_size + profile_dim,
            hidden_dim=hidden_size,
            output_dim=hidden_size,
            dropout=dropout,
        )
        self.router_head = nn.Linear(hidden_size, profile_dim)

        expert_input_dim = hidden_size * 2
        self.expert_heads = nn.ModuleList(
            [
                ExpertResidualHead(
                    input_dim=expert_input_dim,
                    hidden_size=hidden_size,
                    dropout=dropout,
                )
                for _ in range(profile_dim)
            ]
        )
        self.local_gate_grn = GatedResidualNetwork(
            input_dim=expert_input_dim,
            hidden_dim=hidden_size,
            output_dim=hidden_size,
            dropout=dropout,
        )
        self.local_gate_head = nn.Linear(hidden_size, 1)
        self.global_gate_grn = GatedResidualNetwork(
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            output_dim=hidden_size,
            dropout=dropout,
        )
        self.global_gate_head = nn.Linear(hidden_size, 1)

        nn.init.constant_(self.local_gate_head.bias, local_gate_bias_init)
        nn.init.constant_(self.global_gate_head.bias, global_gate_bias_init)

    def forward(
        self,
        encoder_cont: torch.Tensor,
        decoder_known: torch.Tensor,
        profile_prior: torch.Tensor,
        baseline_stats: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoder_selected, encoder_weights = self.encoder_vsn(encoder_cont)
        decoder_selected, decoder_weights = self.decoder_vsn(decoder_known)

        encoder_output, encoder_state = self.encoder_lstm(encoder_selected)
        decoder_output, _ = self.decoder_lstm(decoder_selected, encoder_state)
        sequence = torch.cat([encoder_output, decoder_output], dim=1)
        residual_sequence = torch.cat([encoder_selected, decoder_selected], dim=1)
        sequence = self.post_lstm_gate(sequence, residual_sequence)
        sequence = self.enrichment_grn(sequence)

        query = sequence[:, -self.prediction_length :, :]
        attention_mask = self._build_attention_mask(
            encoder_length=encoder_selected.size(1),
            prediction_length=self.prediction_length,
            device=sequence.device,
        )
        attention_output, attention_weights = self.self_attention(
            query=query,
            key=sequence,
            value=sequence,
            attn_mask=attention_mask,
            need_weights=self.return_attention_weights,
            average_attn_weights=False,
        )
        attention_output = self.post_attention_gate(attention_output, query)
        decoder_features = self.positionwise_grn(attention_output)
        decoder_features = self.pre_output_gate(decoder_features, attention_output)

        context_features = torch.cat(
            [
                encoder_output.mean(dim=1),
                decoder_features.mean(dim=1),
                profile_prior,
                baseline_stats,
            ],
            dim=-1,
        )
        context_vector = self.context_grn(context_features)

        router_input = torch.cat([context_vector, profile_prior], dim=-1)
        router_logits = self.router_head(self.router_grn(router_input))
        router_logits = router_logits + self.router_prior_weight * torch.log(
            profile_prior.clamp_min(1e-6)
        )
        expert_weights = torch.softmax(router_logits, dim=-1)

        expanded_context = context_vector.unsqueeze(1).expand(-1, self.prediction_length, -1)
        expert_input = torch.cat([decoder_features, expanded_context], dim=-1)
        expert_predictions = torch.stack(
            [expert_head(expert_input) for expert_head in self.expert_heads],
            dim=-1,
        )
        mixed_residual = torch.sum(expert_predictions * expert_weights.unsqueeze(1), dim=-1)

        global_gate = torch.sigmoid(
            self.global_gate_head(self.global_gate_grn(context_vector))
        ).squeeze(-1)
        local_gate = torch.sigmoid(
            self.local_gate_head(self.local_gate_grn(expert_input))
        ).squeeze(-1)
        residual_prediction = mixed_residual * global_gate.unsqueeze(-1) * local_gate

        return {
            "residual_prediction": residual_prediction,
            "mixed_residual": mixed_residual,
            "expert_predictions": expert_predictions,
            "expert_weights": expert_weights,
            "profile_prior": profile_prior,
            "global_gate": global_gate,
            "local_gate": local_gate,
            "context_vector": context_vector,
            "encoder_variable_weights": encoder_weights,
            "decoder_variable_weights": decoder_weights,
            "attention_weights": attention_weights if self.return_attention_weights else None,
        }

    @staticmethod
    def _build_attention_mask(
        encoder_length: int,
        prediction_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_length = encoder_length + prediction_length
        key_positions = torch.arange(total_length, device=device)
        query_positions = encoder_length + torch.arange(prediction_length, device=device)
        return key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
