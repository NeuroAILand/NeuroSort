import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelConfig:
    def __init__(self,
                 vocab_size=2048,  # For waveform of spike embedding
                 hidden_size=256,  # Initial linear layer expands dimensions  
                 spike_linear_size=128,  # Initial linear layer expands dimensions for spike
                 num_hidden_layers=1,  # The number of hidden layers
                 num_attention_heads=8,
                 intermediate_size=128,  # Intermediate linear layer expands dimensions  
                 hidden_act="gelu",
                 hidden_dropout_prob=0.0,  # The dropout rate for the outputs of layers other than the attention module
                 attention_probs_dropout_prob=0.1,  # The dropout rate for the attention module
                 num_channels=None,  
                 num_nodes_time=8192 * 16,  # For spike time embedding
                 layer_norm_eps=1e-12,  # The normalization layer prevents the denominator from being zero
                 feature_dim=None,  
                 num_nodes_probe=10,  # Number of probes
                 fc_size=512,  # Dimensionality reduction from hidden_size * feature_dim to mlm_size  
                 tree_depth=None,  # The recommended depth of the tree
                 initial_epochs=0,
                 ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.num_channels = num_channels
        self.num_nodes_time = num_nodes_time
        self.layer_norm_eps = layer_norm_eps
        self.feature_dim = feature_dim
        self.num_nodes_probe = num_nodes_probe
        self.fc_size = fc_size
        self.tree_depth = tree_depth
        self.initial_epochs = initial_epochs
        self.spike_linear_size = hidden_size


class BertForTraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel(self.config)
        self.ln = nn.LayerNorm(self.config.hidden_size * self.config.feature_dim, eps=self.config.layer_norm_eps)
        self.weight_deep = nn.Parameter(torch.ones(1))  
        self.weight_wide = nn.Parameter(torch.ones(1) * 0.2)   

    def forward(self, features, channel, time, epo):
        embedding, outputs, features_1 = self.bert([features, channel, time], epo)

        if epo < self.config.initial_epochs:
            new_out = outputs.view(-1, self.config.hidden_size * self.config.feature_dim)

        else:
            new_out = (outputs + self.weight_wide * features_1.view(features_1.shape[0], 1, features_1.shape[1])).view(-1, self.config.hidden_size * self.config.feature_dim)

        return new_out


class BertMLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        self.fc = nn.Linear(self.config.hidden_size, self.config.mlm_size)

    def forward(self, x):
        x = self.ln(x)
        x = self.fc(x)
        x = F.relu(x)
        return x


class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(self.config)
        self.encoder = BertEncoder(self.config)
        self.pooler = BertPooler(self.config)

        self.spike_projection = nn.Linear(self.config.feature_dim, self.config.spike_linear_size)
        self.spike_conv = nn.Conv1d(in_channels=1, out_channels=self.config.spike_linear_size,
                                    kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.spike_dropout = nn.Dropout(0.01)
        self.Ln = nn.LayerNorm(self.config.spike_linear_size, eps=self.config.layer_norm_eps)

    def forward(self, inputs, epo):
        features, channel, time = inputs
        if epo < self.config.initial_epochs:
            embedding_output = self.embeddings(inputs)
            encoder_outputs = self.encoder(embedding_output)
            # pooled_output = self.pooler(encoder_outputs)
            with torch.no_grad():
                features_1 = self.Ln(self.spike_dropout(self.spike_projection(features.float())))
        else:
            features_1 = self.Ln(self.spike_dropout(self.spike_projection(features.float())))

            embedding_output = self.embeddings(inputs)
            encoder_outputs = self.encoder(embedding_output)
           
        return embedding_output, encoder_outputs, features_1


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super(LearnablePositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(seq_len, d_model)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        return self.position_embeddings(positions)


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_embeddings = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.channel_embeddings = nn.Embedding(self.config.num_channels + 1, self.config.hidden_size)
        self.time_embeddings = nn.Embedding(self.config.num_nodes_time, self.config.hidden_size)
        self.prode_embeddings = nn.Embedding(self.config.num_nodes_probe, self.config.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)


    def forward(self, inputs):
        # # features, channel, time, probe = inputs
        features, channel, time = inputs
        embedded_spike = self.feature_embeddings(features.long())
        embedded_time = self.time_embeddings(time.long())
        embedded_channel = self.channel_embeddings(channel.long())
        # embedded_probe = self.prode_embeddings(probe)
        embeddings = embedded_spike + embedded_time + embedded_channel  #  + embedded_probe
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(config)
        self.intermediate = LayerInterMediate(config)
        self.outputs = BertOutput(config)

    def forward(self, hidden_states):
        attention_output = self.self_attention(hidden_states)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.outputs(intermediate_output, attention_output)
        return layer_output


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_ = Attention(config)
        self.outputs = MHAOut(config)

    def forward(self, hidden_states):
        self_output = self.self_(hidden_states)
        attention_output = self.outputs(self_output, hidden_states)
        return attention_output


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = (x.size(0), x.size(1), self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        q_in = self.query(hidden_states)
        k_in = self.key(hidden_states)
        v_in = self.value(hidden_states)
        q = self.transpose_for_scores(q_in)
        k = self.transpose_for_scores(k_in)
        v = self.transpose_for_scores(v_in)

        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = F.softmax(attention_scores, dim=-1)
        context_layer = torch.matmul(attention_probs, v)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(hidden_states.shape)
        return context_layer


class MHAOut(nn.Module):
    def __init__(self, config):
        super(MHAOut, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, res):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + res)
        return hidden_states


class LayerInterMediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, res):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + res)
        return hidden_states


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class TreeDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, tree_depth, hidden_dropout_prob):
        super(TreeDecoder, self).__init__()
        self.model = ForestModel(input_dim, tree_depth, output_dim,
                                 dropout=hidden_dropout_prob)

    def forward(self, x):
        output = self.model(x)
        return output


class ForestModel(nn.Module):
    def __init__(self, input_width, tree_depth, output_width, activation=nn.ReLU(), dropout=0.0):  
        super(ForestModel, self).__init__()
        self.input_width = input_width  
        self.tree_depth = tree_depth  
        self.output_width = output_width  
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.num_trees = output_width // (2 ** tree_depth)
        self.linear1 = nn.Linear(input_width, self.num_trees)
        self.n_leaves = 2 ** tree_depth
        self.n_nodes = tree_depth * 2  
        l1_init_factor = 1.0 / math.sqrt(input_width)
        self.node_weights = nn.Parameter(
            torch.empty((self.num_trees, self.n_nodes, self.input_width)).uniform_(-l1_init_factor, +l1_init_factor))
        self.node_biases = nn.Parameter(
            torch.empty((self.num_trees, self.n_nodes, 1)).uniform_(-l1_init_factor, +l1_init_factor))

    def forward(self, x):
        batch_size = x.size(0)
        linear_output = self.linear1(x)
        probabilities = F.softmax(linear_output, dim=1)

        node_outputs = self.activation(
            torch.matmul(x, self.node_weights.permute(2, 0, 1).reshape(self.input_width, -1))
            + self.node_biases.reshape(1, -1))

        node_outputs = node_outputs.reshape(batch_size, self.num_trees, self.n_nodes)   

        left_child = 2 * torch.arange(0, self.n_nodes // 2).to(x.device)
        right_child = left_child + 1

        combined = torch.stack((node_outputs[:, :, left_child], node_outputs[:, :, right_child]), dim=-1)
        softmax_combined = self.stable_softmax(combined)
        left_prob = softmax_combined[:, :, :, 0]
        right_prob = softmax_combined[:, :, :, 1]

        current_layer_values = probabilities.unsqueeze(2)  # (batch_size, num_trees, 1)
        for depth in range(self.tree_depth):
            left_values = current_layer_values * left_prob[:, :, depth].unsqueeze(2)
            right_values = current_layer_values * right_prob[:, :, depth].unsqueeze(2)
            current_layer_values = torch.cat([left_values, right_values], dim=2)

        outputs = current_layer_values.view(batch_size, -1)  # (batch_size, num_trees * n_leaves)
        return outputs

    def stable_softmax(self, x):
        shift_x = x - torch.max(x, dim=-1, keepdim=True)[0]
        exp_x = torch.exp(shift_x)
        sum_exp_x = torch.sum(exp_x, dim=-1, keepdim=True)
        softmax_x = exp_x / sum_exp_x
        return softmax_x

    def compute_leaf_nodes(self, root_value, left_child, right_child):
        current_layer_values = root_value.unsqueeze(0)  
        for depth in range(self.tree_depth):
            next_layer_values = []
            for node_value in current_layer_values:
                left_value = node_value * left_child[depth]
                right_value = node_value * right_child[depth]
                next_layer_values.append(left_value.unsqueeze(0))  
                next_layer_values.append(right_value.unsqueeze(0))  
            current_layer_values = torch.cat(next_layer_values, dim=0)  
        return current_layer_values


def float_to_int(f_input, range, info=None):
    q_min = 0
    q_max = range - 1  

    if info is not None:
        r_max = torch.tensor(info[0])
        r_min = torch.tensor(info[1])
    else:
        r_max = torch.max(f_input)
        r_min = torch.min(f_input)

    scale = (r_max - r_min) / (q_max - q_min)
    scale = torch.clamp(scale, min=1e-9)  

    zero_point = q_min - torch.round(r_min / scale).to(torch.long)
    zero_point = torch.clamp(zero_point, q_min, q_max)

    q_x = (torch.round(f_input / scale) + zero_point).to(torch.long)
    q_x = torch.clamp(q_x, q_min, q_max)
    return q_x


def normalize_to_01_dim(tensor, dim):
    min_val, _ = tensor.min(dim, keepdim=True)
    max_val, _ = tensor.max(dim, keepdim=True)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor


def standardize_dim(tensor, dim):
    mean = tensor.mean(dim, keepdim=True)
    std = tensor.std(dim, keepdim=True)
    standardized_tensor = (tensor - mean) / std
    return standardized_tensor

