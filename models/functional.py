import torch.nn.functional as F
import torch.nn as nn
import torch


class CustomTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, do_sa=False):
        x = tgt
        if self.norm_first:
            if do_sa:
                x = x + self._sa_block(
                    self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x_residue, attention = self._mha_block(
                self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + x_residue
            x = x + self._ff_block(self.norm3(x))
        else:
            # default branch
            if do_sa:
                x = self.norm1(
                    x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x_residue, attention = self._mha_block(
                x, memory, memory_mask, memory_key_padding_mask)
            x = self.norm2(x + x_residue)
            x = self.norm3(x + self._ff_block(x))

        return x, attention

    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        x, attention = self.multihead_attn(
            x, mem, mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True, average_attn_weights=False)
        return self.dropout2(x), attention


def matrix_mul(inputs, weight, bias=False):
    feature_list = []
    for feature in inputs:
        feature = torch.mm(feature, weight)
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)

    return torch.cat(feature_list, 0)


def element_wise_mul(input1, input2):
    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 0).unsqueeze(0)


class SentAttNet(nn.Module):
    def __init__(self, sent_hidden_size=50, word_hidden_size=50,
                 num_classes=14):
        super(SentAttNet, self).__init__()

        self.sent_weight = nn.Parameter(torch.zeros(
            2 * sent_hidden_size, 2 * sent_hidden_size))
        self.sent_bias = nn.Parameter(torch.zeros(1, 2 * sent_hidden_size))
        self.context_weight = nn.Parameter(
            torch.zeros(2 * sent_hidden_size, 1))

        self.gru = nn.GRU(2 * word_hidden_size,
                          sent_hidden_size, bidirectional=True)
        self.fc = nn.Linear(2 * sent_hidden_size, num_classes)
        # self.sent_softmax = nn.Softmax()
        # self.fc_softmax = nn.Softmax()
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.sent_weight.data.normal_(mean, std)
        self.sent_bias.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, inputs, hidden_state):

        f_output, h_output = self.gru(inputs, hidden_state)
        output = matrix_mul(f_output, self.sent_weight, self.sent_bias)
        output = matrix_mul(output, self.context_weight
                            ).squeeze(-1).permute(1, 0)
        output = F.softmax(output)
        output = element_wise_mul(f_output, output.permute(1, 0)).squeeze(0)
        output = self.fc(output)

        return output, h_output
