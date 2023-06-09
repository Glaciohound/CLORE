from dataclasses import dataclass
from typing import Optional, Tuple
from contextlib import nullcontext

from copy import deepcopy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers.file_utils import ModelOutput
from models.functional import matrix_mul, element_wise_mul


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1,
                 max_length=256):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        # self.max_length = max_length

        # self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.attn = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        )
        self.attn_key = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.attn_combine = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        )
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        # TODO
        # self.gru = nn.LSTM(self.hidden_size, self.hidden_size, 2)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.out_to_in = nn.Sequential(
            nn.Linear(self.output_size, 2*self.output_size),
            nn.LeakyReLU(),
            nn.Linear(2*self.output_size, self.hidden_size)
        )

    def forward(self, last_tok, hidden, encoder_outputs):
        # embedded = self.embedding(input).view(1, 1, -1)
        # embedded = last_tok.view(1, 1, -1)
        embedded = self.dropout(last_tok)

        # attn_weights = F.softmax(
        #     self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_query = self.attn(torch.cat((embedded[0], hidden[0]), 1))
        attn_weights = F.softmax(
            (attn_query[None] * self.attn_key(encoder_outputs)).sum(-1),
            dim=0
        )
        attn_applied = torch.bmm(attn_weights.transpose(1, 0).unsqueeze(1),
                                 encoder_outputs.transpose(1, 0)).squeeze(1)

        output = torch.cat((embedded[0], attn_applied), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        # output = F.log_softmax(self.out(output[0]), dim=1)
        output = self.out(output[0])
        return output, hidden, attn_weights

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

    def decode(self, inputs, contexts, max_length):
        outputs = []
        attentions = []
        batch_size = contexts.shape[1]
        hidden = self.initHidden(batch_size).to(inputs)
        for i in range(max_length):
            output, hidden, attn = self.forward(inputs, hidden, contexts)
            outputs.append(output)
            attentions.append(attn)
            inputs = self.out_to_in(output).unsqueeze(0)
        outputs = torch.stack(outputs, 0)
        attentions = torch.stack(attentions, 0)
        return outputs, attentions


class ProgramExecutor(nn.Module):
    def __init__(self, hidden_size, mask_out_templates):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_type_template = 3
        final_size = self.hidden_size + self.n_type_template + 2
        self.program_final_layer = nn.Linear(self.hidden_size, final_size)
        self.program_length = 5
        self.mask_out_templates = mask_out_templates

    def execute_program(self, inputs, program_vec, program_key,
                        program_query,
                        program_type, program_bias, program_scale):
        # inputs shape: (batch_size, length, dim)
        # programs vectors shape: (steps, n_guideline, program_dim)
        # programs bias shape: (steps, n_guideline)
        # device = inputs.device
        batch_size, length, dim = inputs.shape
        n_step, n_program, _, _ = program_vec.shape
        n_type_template = program_type.shape[2]
        attention_logits = torch.zeros(
            0, batch_size, n_program, n_type_template
        ).to(inputs)
        sigma = math.sqrt(dim) * 6
        select_logits_all = []
        # gamma = 0.85
        # tau = 0.25
        for step_i, (_vec, _, _query, _type) in enumerate(
            zip(program_vec, program_key,
                program_query, program_type)):
            # calculating attention logits of each program type
            # per-step programs shape: (n_guideline, program_dim)

            select_logits_sent = (
                torch.matmul(inputs, _vec[:, 0].transpose(1, 0))/sigma
            )
            select_logits_all.append(select_logits_sent)
            select_logits = select_logits_sent.max(1)[0].unsqueeze(-1)

            arg1 = F.softmax(
                (program_key[:step_i] * _query[None]).sum(-1)/sigma, dim=0
            )
            # shape: (i-1, n_guideline)
            arg1_logits = (arg1[:, None] * attention_logits).sum(0)
            # shape: (bs, n_guideline)

            if step_i != 0:
                arg2_logits = attention_logits[-1]
            else:
                arg2_logits = arg1_logits

            and_logits = torch.min(arg1_logits, arg2_logits)
            or_logits = torch.max(arg1_logits, arg2_logits)
            not_logits = - arg2_logits
            copy_logits = arg2_logits
            deduce_logits = torch.max(-arg1_logits, arg2_logits)
            # shape: (bs, n_guideline)

            final_logits = (
                _type[None] *
                torch.stack([
                    select_logits.expand_as(and_logits), and_logits, or_logits,
                    not_logits, copy_logits, deduce_logits,
                ], 3)
            ).sum(3)
            attention_logits = torch.cat([
                attention_logits, final_logits[None]], 0)

        output_scores = final_logits * program_scale[0] + program_bias[0]
        # shape: (bs, n_guideline)
        return output_scores, torch.stack(select_logits_all)

    def forward(self, inputs, program):
        program = self.program_final_layer(program)
        program_vec = program[:, :, None, :self.hidden_size]
        # program_key = program[:, :, self.hidden_size:self.hidden_size*2]
        # program_query = program[:, :, self.hidden_size*2:self.hidden_size*3]
        # program_type = program[:, :, self.hidden_size*3:-1]
        program_type_index_logits = program[0, :, self.hidden_size:-2]
        if len(self.mask_out_templates) != 0:
            program_type_index_logits[:, self.mask_out_templates] = -100
        program_type_index = F.softmax(
            program_type_index_logits, dim=1
        )
        # shape: n_program, n_type_template
        program_bias = program[:, :, None, -2]
        program_scale = program[:, :, None, -1]
        # shape: n_step, n_program, n_type_template, dim

        # templated programs
        # n_program = program.shape[1]
        program_length = self.program_length
        n_type = 6
        n_type_template = self.n_type_template
        program_templates = torch.ones(
            n_type_template, program_length, dtype=int, device=program.device
        ) * 4
        program_templates[:, 0] = 0
        program_templates[1:, 1] = 0
        program_templates[1:, 2] = 1
        program_templates[2, 3] = 0
        program_templates[2, 4] = 1
        program_templates = F.one_hot(program_templates, n_type).float()
        program_templates = program_templates.transpose(0, 1)[:, None]
        program_key = torch.zeros(
            program_length, 1, 1, self.hidden_size
        ).to(program)
        program_key[:, 0, 0, :program_length] = torch.eye(
            program_length
        )
        program_template_query = torch.zeros(
            program_length, 1, n_type_template, self.hidden_size
        ).to(program)
        program_template_query[2, :, 1:, 0] = 1000
        program_template_query[4, :, 2, 2] = 1000

        scores, reason_attentions = self.execute_program(
            inputs, program_vec,
            program_key, program_template_query,
            program_templates, program_bias, program_scale)
        # scores shape: batch_size, n_program, n_type_template
        scores = (scores * program_type_index).sum(-1)

        return scores, reason_attentions, \
            (program_vec, program_bias, program_scale, program_type_index)
