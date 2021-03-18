import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
# Due to backwards compatibility we need to keep the below structure for mapping RNN type
from omegaconf import OmegaConf

from deepspeech_pytorch.configs.train_config import SpectConfig
from deepspeech_pytorch.enums import SpectrogramWindow

supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU
}
supported_rnns_inv = dict((v, k) for k, v in supported_rnns.items())


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)#còn lại 2 chieeuf
        x = self.module(x)#hàm này thay đổi x
        x = x.view(t, n, -1)# 3 chiều
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class MaskConv(nn.Module):
    def __init__(self, seq_module):
        """
        Adds padding to the output of the module based on the given lengths. This is to ensure that the
        results of the model do not change when batch sizes change during inference.
        Input needs to be in the shape of (BxCxDxT)
        :param seq_module: The sequential module containing the conv stack.
        """
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        """
        :param x: The input of size BxCxDxT
        :param lengths: The actual length of each sequence in the batch
        :return: Masked output from the module
        """
        for module in self.seq_module:#self.seq_module= Conv2d, BatchNorm, Hardtanh, Conv2d, BatchNorm, Hardtanh
            x = module(x)#//áp dụng từng lớp lên inputs, x sau khi áp dụng lớp Conv2d bị đổi số chiều [32, 32, 81, 350]
            mask = torch.BoolTensor(x.size()).fill_(0)# tạo ma trận bool các giá trị False ... tương ứng với số chiều của x
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(lengths):# kiểm tra từng độ dài chuỗi văn bản kết quả
                length = length.item()# lấy từng độ dài chuỗi văn bản
                if (mask[i].size(2) - length) > 0:# có chuỗi nào < mask[i].size(2)
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)# không thay đổi số của input x
        return x, lengths


class InferenceBatchSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            return F.softmax(input_, dim=-1)
        else:
            return input_


class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=True)#GRU(1312, 1600, bidirectional=True)
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, output_lengths):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths)#PackedSequence(data=tensor([[0.4077, 0.4731, 0.2859,  ..., 0.7847, 0.7739, 0.5264] device='cuda:0', dtype=torch.float16,grad_fn=<PackPaddedSequenceBackward>), batch_sizes=tensor([32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, .....,1,  1,  1,  1,  1,  1,  1,  1]), sorted_indices=None, unsorted_indices=None)
        x, h = self.rnn(x)#PackedSequence(data=tensor([[ 0.0755,  0.0405,  0.0306,  ..., -0.0295,  0.1142,  0.0401],...
        #h=#tensor([[[ 8.9539e-02,  9.7778e-02, -7.5989e-02,  ..., -1.1945e-01,.., device='cuda:0', dtype=torch.float16,grad_fn=<CudnnRnnBackward>)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)# ma trận batch_size phía sau là 0000
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # tensor([[[-5.8533e-02,  2.2675e-02, -3.0853e-02,  ..., -4.0344e-02, (TxNxH*2) -> (TxNxH) by sum
        return x


class Lookahead(nn.Module):
    # Wang et al 2016 - Lookahead Convolution Layer for Unidirectional Recurrent Neural Networks
    # input shape - sequence, batch, feature - TxNxH
    # output shape - same as input
    def __init__(self, n_features, context):
        super(Lookahead, self).__init__()
        assert context > 0
        self.context = context
        self.n_features = n_features
        self.pad = (0, self.context - 1)
        self.conv = nn.Conv1d(self.n_features, self.n_features, kernel_size=self.context, stride=1,
                              groups=self.n_features, padding=0, bias=None)

    def forward(self, x):
        x = x.transpose(0, 1).transpose(1, 2)
        x = F.pad(x, pad=self.pad, value=0)
        x = self.conv(x)
        x = x.transpose(1, 2).transpose(0, 1).contiguous()
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'n_features=' + str(self.n_features) \
               + ', context=' + str(self.context) + ')'


class DeepSpeech(nn.Module):
    def __init__(self, rnn_type, labels, rnn_hidden_size, nb_layers, audio_conf,
                 bidirectional, context=20):
        super(DeepSpeech, self).__init__()

        self.hidden_size = rnn_hidden_size
        self.hidden_layers = nb_layers
        self.rnn_type = rnn_type
        self.audio_conf = audio_conf
        self.labels = labels
        self.bidirectional = bidirectional

        sample_rate = self.audio_conf.sample_rate
        window_size = self.audio_conf.window_size
        num_classes = len(self.labels)
        #Conv2d chuyển đổi chanel in và out
        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        ))
        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        rnn_input_size = int(math.floor((sample_rate * window_size) / 2) + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
        rnn_input_size *= 32

        rnns = []#, dòng dưới, là đầu tiên đưa phổ vào mô hình, nên input_size=rnn_input_size
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                       bidirectional=bidirectional, batch_norm=False)#có batch norm bên trong nếu truyền tham số batch_norm
        rnns.append(('0', rnn))#Lớp đầu tiên ko có chuẩn hóa
        for x in range(nb_layers - 1):
            rnn = BatchRNN(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                           bidirectional=bidirectional)
            rnns.append(('%d' % (x + 1), rnn))
        self.rnns = nn.Sequential(OrderedDict(rnns))#nn.Sequential Thêm các key '1', '2','3' cho các lớp rnn
        self.lookahead = nn.Sequential(
            # consider adding batch norm?
            Lookahead(rnn_hidden_size, context=context),
            nn.Hardtanh(0, 20, inplace=True)
        ) if not bidirectional else None

        fully_connected = nn.Sequential(
            nn.BatchNorm1d(rnn_hidden_size),
            nn.Linear(rnn_hidden_size, num_classes, bias=False)#áp dụng chuyển đổi y=x*A^T+b => chuyển đầu ra từ 1600 thành số kí tự của bảng chữ cái
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        self.inference_softmax = InferenceBatchSoftmax()#nếu là testing thì sẽ chuyển đổi ma trận kq thành ma trận xác suất, tổng 1 dòng =1

    def forward(self, x, lengths):#x là inputs(ma trận 32 mẫu, chứa phổ 2 chiều), lengths là độ dài thực sự từng phổ (từng câu nói)
        lengths = lengths.cpu().int()#length bị đổi giá trị #tensor([488601, 476718, 458544, 391440, 386547, 361383, 359286, 350898, 324336,...] #mỗi giá trị length ban đầu tự nhân với độ dài câu nói lớn nhất
        output_lengths = self.get_seq_lens(lengths)# tính độ dài văn bản kết quả với từng độ dài câu nói tensor([244301, 238359, 229272, 195720, 193274, 180692, 179643, 175449, 162168,..)
        x, _ = self.conv(x, output_lengths)# //LỚP NÀY XÁC ĐỊNH ĐƯỢC ĐỘ DÀI VĂN BẢN ĐẦU RA, đưa qua lớp tích chập Conv2d, BatchNorm, Hardtanh, Conv2d, BatchNorm, Hardtanh, trả về x

        sizes = x.size()#torch.Size([32, 32, 41, 350])
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension, bỏ bớt số chiều (32,1312,350)
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH, đổi giá trị x

        for rnn in self.rnns:#đi qua 7 lớp GRU
            x = rnn(x, output_lengths)

        if not self.bidirectional:  # no need for lookahead layer in bidirectional
            x = self.lookahead(x)

        x = self.fc(x)#size x torch.Size([127, 1, 93])
        x = x.transpose(0, 1)#3 chiều, torch.Size([1, 127, 93])
        # identity in training mode, softmax in eval mode
        x = self.inference_softmax(x)
        return x, output_lengths

    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor      ;; param input_length: 1D Tensor
        :return: 1D Tensor scaled by model  ;; return: 1D Tensor được chia tỷ lệ theo mô hình
        """
        seq_len = input_length
        for m in self.conv.modules():#self.conv.modules() gồm 1 lớp Mask, 1 lớp Sequencial,  Conv2d, BatchNorm, Hardtanh, Conv2d, BatchNorm, Hardtanh
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)
        return seq_len.int()# nếu m là Conv2d thì có kiểu Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5))

    @classmethod
    def load_model(cls, path):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = DeepSpeech.load_model_package(package)
        return model

    @classmethod
    def load_model_package(cls, package):
        # TODO Added for backwards compatibility, should be remove for new release
        if OmegaConf.get_type(package['audio_conf']) == dict:
            audio_conf = package['audio_conf']
            package['audio_conf'] = SpectConfig(sample_rate=audio_conf['sample_rate'],
                                                window_size=audio_conf['window_size'],
                                                window=SpectrogramWindow(audio_conf['window']))
        model = cls(rnn_hidden_size=package['hidden_size'],
                    nb_layers=package['hidden_layers'],
                    labels=package['labels'],
                    audio_conf=package['audio_conf'],
                    rnn_type=supported_rnns[package['rnn_type']],
                    bidirectional=package.get('bidirectional', True))
        model.load_state_dict(package['state_dict'])
        return model

    def serialize_state(self):
        return {
            'hidden_size': self.hidden_size,
            'hidden_layers': self.hidden_layers,
            'rnn_type': supported_rnns_inv.get(self.rnn_type, self.rnn_type.__name__.lower()),
            'audio_conf': self.audio_conf,
            'labels': self.labels,
            'state_dict': self.state_dict(),
            'bidirectional': self.bidirectional,
        }

    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params
