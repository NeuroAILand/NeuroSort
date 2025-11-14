import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from AttenModel import BertForTraining, float_to_int, TreeDecoder
from torch.amp import GradScaler, autocast
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


def set_seed(seed_value):
    # random.seed(seed_value)
    # np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def iic_loss(prob_x, prob_qx):
    repeat_factor = int(prob_qx.shape[1])
    repeated_prob_x = prob_x.unsqueeze(1).expand(-1, repeat_factor, -1)
    repeated_prob_x = repeated_prob_x.reshape(-1, repeated_prob_x.shape[-1])
    transposed_prob_x = repeated_prob_x.t()

    prob_qx = prob_qx.reshape(-1, prob_qx.shape[-1])
    joint_prob_matrix = torch.mm(transposed_prob_x, prob_qx)
    joint_prob_matrix = (joint_prob_matrix + joint_prob_matrix.t()) / 2  
 

    total_joint_prob = joint_prob_matrix.sum()
    normalized_joint_prob = joint_prob_matrix / total_joint_prob
    normalized_joint_prob = torch.clamp(normalized_joint_prob, min=1e-6)

    marginal_prob_x = normalized_joint_prob.sum(dim=1, keepdim=True)  
    marginal_prob_qx = normalized_joint_prob.sum(dim=0, keepdim=True)  

    marginal_prob_x = torch.clamp(marginal_prob_x, min=1e-6)
    marginal_prob_qx = torch.clamp(marginal_prob_qx, min=1e-6)

    mutual_information_loss = -torch.sum(normalized_joint_prob * (
            torch.log(normalized_joint_prob) - torch.log(marginal_prob_x) - torch.log(marginal_prob_qx)))

    return mutual_information_loss


def find_consecutive_indices(data):
    element_indices = {}
    for i, value in enumerate(data):
        if value not in element_indices:
            element_indices[value] = []
        element_indices[value].append(i)

    sorted_elements = sorted(element_indices.keys())

    result = []
    current_sequence = element_indices[sorted_elements[0]]
    for i in range(1, len(sorted_elements)):
        if sorted_elements[i] <= sorted_elements[i - 1] + 2:
            current_sequence.extend(element_indices[sorted_elements[i]])
        else:
            result.append(current_sequence)
            current_sequence = element_indices[sorted_elements[i]]
    result.append(current_sequence)
    return result


def find_indices(arr):
    indices_dict = {}
    for index, value in enumerate(arr):
        if value not in indices_dict:
            indices_dict[value] = []
        indices_dict[value].append(index)

    return list(indices_dict.values())


class CustomDataset(Dataset):
    def __init__(self, spike, channel, time, config, transform=None):
        self.spike = spike
        self.channel = channel
        self.time = time
        self.transform = transform
        self.config = config

    def __len__(self):
        return len(self.spike)

    def __getitem__(self, idx):
        spike = self.spike[idx]
        channel = self.channel[idx]
        time = self.time[idx]

        if self.transform:
            spike = self.transform(spike, self.config)
        return spike, channel, time
    

class ModelEncoder_inference(nn.Module):
    def __init__(self, config, independent_rng, is_electrode_correlation, spike_info, time_info):
        super(ModelEncoder_inference, self).__init__()
        self.config = config
        if is_electrode_correlation:
            self.config.num_channels = self.config.num_channels + 1
        self.encoder = BertForTraining(self.config)
        self.independent_rng = independent_rng
        self.is_electrode_correlation = is_electrode_correlation
        self.spike_info = spike_info
        self.time_info = time_info

    def forward(self, data0, channel, time, epo):
        data_all = float_to_int(data0, self.config.vocab_size, self.spike_info)
        time = float_to_int(time, self.config.num_nodes_time, self.time_info)
        data00 = self.encoder(data_all, channel, time, epo)
        return data00


class ModelEncoder(nn.Module):
    def __init__(self, config, independent_rng, is_electrode_correlation, spike_info, time_info):
        super(ModelEncoder, self).__init__()
        self.config = config
        if is_electrode_correlation:
            self.config.num_channels = self.config.num_channels + 1
        self.encoder = BertForTraining(self.config)
        self.independent_rng = independent_rng
        self.is_electrode_correlation = is_electrode_correlation
        self.spike_info = spike_info
        self.time_info = time_info

    def forward(self, data_all, channel, time, epo):
        data_all = float_to_int(data_all, self.config.vocab_size, self.spike_info)
        data0 = data_all[:, 0, :]
        data1 = data_all[:, 1, :]
        data2 = data_all[:, 2, :]
        data3 = data_all[:, 3, :]
        data4 = data_all[:, 4, :]
        data5 = data_all[:, 5, :]
        data_amp = data_all[:, 6, :]

        channel = channel.reshape(-1, 1)
        time = float_to_int(time, self.config.num_nodes_time, self.time_info)
        time = time.reshape(-1, 1)

        rand_int_time = torch.tensor(
            [
                torch.randint(int(low.item()) + self.config.feature_dim, self.config.num_nodes_time, (6,),
                              generator=self.independent_rng).tolist()
                if int(low.item()) + self.config.feature_dim < self.config.num_nodes_time else
                [self.config.num_nodes_time - 1] * 6
                for low in time
            ]
        ).view(-1, time.shape[0]).to('cuda')
        data00 = self.encoder(data0, channel, time, epo)
        if self.is_electrode_correlation:
            rand_floats = torch.rand((channel.shape[0], 5), generator=self.independent_rng) * 2 - 1
            rand_ints = torch.where(rand_floats >= 0, torch.tensor(2), torch.tensor(1)).to('cuda')
            data11_amp = self.encoder(data_amp, channel + rand_ints[:, 0].view(-1, 1), time, epo)
            rand_int = torch.randint(0, 3, (5,), generator=self.independent_rng).to('cuda')
            data11 = self.encoder(data1, channel + rand_int[0], rand_int_time[0].view(-1, 1), epo)
            data22 = self.encoder(data2, channel + rand_int[1], rand_int_time[1].view(-1, 1), epo)
            data33 = self.encoder(data3, channel + rand_int[2], rand_int_time[2].view(-1, 1), epo)
            data44 = self.encoder(data4, channel + rand_int[3], rand_int_time[3].view(-1, 1), epo)
            data55 = self.encoder(data5, channel + rand_int[4], rand_int_time[4].view(-1, 1), epo)
            data000 = self.encoder(data0, channel, rand_int_time[5].view(-1, 1), epo)
            aug_data = torch.stack([data11, data22, data33, data44, data55, data11_amp, data000], dim=1)
        
        else:
            data11 = self.encoder(data1, channel, rand_int_time[0].view(-1, 1), epo)
            data22 = self.encoder(data2, channel, rand_int_time[1].view(-1, 1), epo)
            data33 = self.encoder(data3, channel, rand_int_time[2].view(-1, 1), epo)
            data55 = self.encoder(data5, channel, rand_int_time[4].view(-1, 1), epo)
            aug_data = torch.stack([data11, data22, data33, data55], dim=1)

        return data00, aug_data


class ModelDecoder(nn.Module):
    def __init__(self, config, output_dim):
        super(ModelDecoder, self).__init__()
        self.output_dim = output_dim
        self.input_dim = config.hidden_size * config.feature_dim       
        self.tree_depth = config.tree_depth
        self.decoder = TreeDecoder(self.input_dim, self.output_dim, self.tree_depth, config.hidden_dropout_prob)

    def forward(self, data):
        if len(data.shape) == 2:
            x_main = self.decoder(data)
            return x_main
        elif len(data.shape) == 3:
            x_main_list = []
            for i in range(data.shape[1]):
                decoded = self.decoder(data[:, i, :])
                x_main_list.append(decoded)
            x_main = torch.stack(x_main_list, dim=1)
            return x_main


class SpikeSort:
    def __init__(self, params):
        self.epoch = params.get('epoch')
        self.patience = params.get('patience')
        self.lossVariance_threshold = params.get('lossVariance_threshold')
        self.params = params
        self.max_num_units = = 1 << (params.get('num_channels') - 1).bit_length()

    def spikeSorting(self, encoder, decoder, encoder_optimizer, decoder_optimizer,
                     train_dataloader, test_dataloader):
        encoder.train()
        decoder.train()
        scaler = GradScaler('cuda')
        losses = []
        for e in range(self.epoch):
            epoch_losses = []
            for batch_idx, (spike, channel, time) in enumerate(train_dataloader):
                spike, channel, time = spike.to('cuda'), channel.to('cuda'), time.to('cuda')
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                with autocast('cuda'):  
                    raw, aug = encoder(spike, channel, time, e)
                    p_raw = decoder(raw)
                    p_aug = decoder(aug)
                    loss = iic_loss(p_raw, p_aug)
                
                scaler.scale(loss).backward()
                scaler.step(encoder_optimizer)
                scaler.step(decoder_optimizer)
                scaler.update()
                epoch_losses.append(loss.item())
            epoch_loss = np.mean(epoch_losses)
            del epoch_losses
            losses.append(epoch_loss)
            print(f'Epoch [{e + 1}/{self.epoch}], MI: {epoch_loss * -1}')

            if len(losses) >= 2 and epoch_loss > losses[-2]:
                break

            if len(losses) >= self.patience:
                variance = np.var(losses[-self.patience:])
                if variance < self.lossVariance_threshold:
                    break
        mi_list = list(map(lambda x: -x, losses))
        del losses
        print('Stop training model and start clustering.')
        p_raw_all = []
        encoder.eval()
        decoder.eval()
        # inference the groups
        with torch.no_grad():  
            for batch_spike, batch_channel, batch_time in test_dataloader:
                batch_spike, batch_channel, batch_time = batch_spike.to('cuda'), batch_channel.to(
                    'cuda'), batch_time.to('cuda')
                raw, aug = encoder(batch_spike, batch_channel, batch_time, -1)
                p_raw = decoder(raw)
                p_indices = torch.argmax(p_raw, dim=1)
                p_raw_all.append(p_indices)

            p_raw_all = torch.cat(p_raw_all, dim=0).cpu()
        if self.inference:
            return p_raw_all_infer, p_raw_all
        else:
            return p_raw_all  

    def getFeature(self, encoder, test_dataloader, seed_group=None):
        encoder.eval()
        if seed_group is None:
            raw_list = []
            with torch.no_grad():  
                for batch_spike, batch_channel, batch_time in test_dataloader:
                    batch_spike, batch_channel, batch_time = batch_spike.to('cuda'), batch_channel.to(
                        'cuda'), batch_time.to('cuda')
                    raw, aug = encoder(batch_spike, batch_channel, batch_time)
                    raw_list.append(raw.cpu().detach())
            raw_out = torch.cat(raw_list, dim=0)
            return raw_out

        else:
            with torch.no_grad():  
                for batch_spike, batch_channel, batch_time in test_dataloader:
                    batch_spike, batch_channel, batch_time = batch_spike.to('cuda'), batch_channel.to(
                        'cuda'), batch_time.to('cuda')
                    raw, aug = encoder(batch_spike, batch_channel, batch_time)
                    raw_cpu = raw.cpu().detach().numpy()
                    if 'feature' in seed_group:
                        
                        dataset = seed_group['feature']
                        
                        dataset.resize((dataset.shape[0] + raw_cpu.shape[0], raw_cpu.shape[1]))
                        
                        dataset[-raw_cpu.shape[0]:] = raw_cpu
                    else:
                        
                        maxshape = (None, raw_cpu.shape[1])  
                        seed_group.create_dataset('feature', data=raw_cpu, maxshape=maxshape)

    def spikeSorting_onlyTrees(self, encoder, decoder, decoder_optimizer,
                               train_dataloader, test_dataloader):
        encoder.eval()
        decoder.train()
        scaler = GradScaler('cuda')
        losses = []
        for e in range(self.epoch):
            epoch_losses = []
            for batch_idx, (spike, channel, time) in enumerate(train_dataloader):
                spike, channel, time = spike.to('cuda'), channel.to('cuda'), time.to('cuda')
                decoder_optimizer.zero_grad()

                with autocast('cuda'):  
                    raw, aug = encoder(spike, channel, time)
                    p_raw = decoder(raw.detach())
                    p_aug = decoder(aug.detach())
                    loss = iic_loss(p_raw, p_aug)
                
                scaler.scale(loss).backward()
                scaler.step(decoder_optimizer)
                scaler.update()
                epoch_losses.append(loss.item())
            epoch_loss = np.mean(epoch_losses)
            del epoch_losses
            losses.append(epoch_loss)
            print(f'Epoch [{e + 1}/{self.epoch}], MI: {epoch_loss * -1}')

            if len(losses) >= 2 and epoch_loss > losses[-2]:
                break

            if len(losses) >= self.patience:
                variance = np.var(losses[-self.patience:])
                if variance < self.lossVariance_threshold:
                    break
        mi_list = list(map(lambda x: -x, losses))
        del losses
        print('Stop training model and start clustering.')
        p_raw_all = []
        decoder.eval()
        with torch.no_grad():  
            all_raw_outputs = []
            for batch_spike, batch_channel, batch_time in test_dataloader:
                batch_spike, batch_channel, batch_time = batch_spike.to('cuda'), batch_channel.to(
                    'cuda'), batch_time.to('cuda')
                raw, aug = encoder(batch_spike, batch_channel, batch_time)
                all_raw_outputs.append(raw.cpu().detach())  # Move the raw to the CPU and detach it from compute graph
                p_raw = decoder(raw)
                p_indices = torch.argmax(p_raw, dim=1)
                p_raw_all.append(p_indices)

            p_raw_all = torch.cat(p_raw_all, dim=0).cpu()
            raw_out = torch.cat(all_raw_outputs, dim=0)
        return mi_list, p_raw_all, raw_out

    def further_check(self, raw_label, electrode):
        label = self.max_num_units
        if self.params.get('is_electrode_correlation'):
            for i in range(self.max_num_units):
                indices = np.where(raw_label == i)[0]
                new_indices = find_consecutive_indices(electrode[indices])
                if len(new_indices) > 1:
                    for j in range(len(new_indices) - 1):
                        raw_label[indices[new_indices[j + 1]]] = label
                        label += 1
        else:
            for i in range(self.max_num_units):
                indices = np.where(raw_label == i)[0]
                new_indices = find_indices(electrode[indices].reshape(-1))
                if len(new_indices) > 1:
                    for j in range(len(new_indices) - 1):
                        raw_label[indices[new_indices[j + 1]]] = label
                        label += 1
        unique_elements, inverse_indices = torch.unique(raw_label, sorted=True,
                                                        return_inverse=True)
        return raw_label, unique_elements, inverse_indices
