import math
import os
from tempfile import NamedTemporaryFile

import librosa
import numpy as np
import soundfile as sf
import sox
import torch
from torch.utils.data import Dataset, Sampler, DistributedSampler, DataLoader

from deepspeech_pytorch.configs.train_config import SpectConfig, AugmentationConfig
from deepspeech_pytorch.loader.spec_augment import spec_augment, visualization_spectrogram

####
import matplotlib.pyplot as plt
import librosa.display
####

def load_audio(path):
    sound, sample_rate = sf.read(path)
    # TODO this should be 32768.0 to get twos-complement range.
    # TODO the difference is negligible but should be fixed for new models.
    #sound = sound.astype('float32') / 32767  # normalize audio
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average
    return sound


class AudioParser(object):
    def parse_transcript(self, transcript_path):
        """
        :param transcript_path: Path where transcript is stored from the manifest file
        :return: Transcript in training/testing format
        """
        raise NotImplementedError

    def parse_audio(self, audio_path):
        """
        :param audio_path: Path where audio is stored from the manifest file
        :return: Audio in training/testing format
        """
        raise NotImplementedError


class NoiseInjection(object):
    def __init__(self,
                 path=None,
                 sample_rate=16000,
                 noise_levels=(0, 0.5)):
        """
        Adds noise to an input signal with specific SNR. Higher the noise level, the more noise added.
        Modified code from https://github.com/willfrey/audio/blob/master/torchaudio/transforms.py
        """
        if not os.path.exists(path):
            print("Directory doesn't exist: {}".format(path))
            raise IOError
        self.paths = path is not None and librosa.util.find_files(path)
        self.sample_rate = sample_rate
        self.noise_levels = noise_levels

    def inject_noise(self, data):
        noise_path = np.random.choice(self.paths)
        noise_level = np.random.uniform(*self.noise_levels)
        return self.inject_noise_sample(data, noise_path, noise_level)

    def inject_noise_sample(self, data, noise_path, noise_level):
        noise_len = sox.file_info.duration(noise_path)
        data_len = len(data) / self.sample_rate
        noise_start = np.random.rand() * (noise_len - data_len)
        noise_end = noise_start + data_len
        noise_dst = audio_with_sox(noise_path, self.sample_rate, noise_start, noise_end)
        assert len(data) == len(noise_dst)
        noise_energy = np.sqrt(noise_dst.dot(noise_dst) / noise_dst.size)
        data_energy = np.sqrt(data.dot(data) / data.size)
        data += noise_level * noise_dst * data_energy / noise_energy
        return data


class SpectrogramParser(AudioParser):
    def __init__(self,
                 audio_conf: SpectConfig,
                 normalize: bool = False,
                 augmentation_conf: AugmentationConfig = None):
        """
        Parses audio file into spectrogram with optional normalization and various augmentations
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param normalize(default False):  Apply standard mean and deviation normalization to audio tensor
        :param augmentation_conf(Optional): Config containing the augmentation parameters
        """
        super(SpectrogramParser, self).__init__()
        self.window_stride = audio_conf.window_stride
        self.window_size = audio_conf.window_size
        self.sample_rate = audio_conf.sample_rate
        self.window = audio_conf.window.value
        self.normalize = normalize
        self.aug_conf = augmentation_conf
        if augmentation_conf and augmentation_conf.noise_dir:
            self.noise_injector = NoiseInjection(path=augmentation_conf.noise_dir,
                                                 sample_rate=self.sample_rate,
                                                 noise_levels=augmentation_conf.noise_levels)
        else:
            self.noise_injector = None

    def parse_audio(self, audio_path):#t???o ph??? cho ?????u v??o ??m thanh
        if self.aug_conf and self.aug_conf.speed_volume_perturb:
            y = load_randomly_augmented_audio(audio_path, self.sample_rate)
        else:
            y = load_audio(audio_path)#th?? vi???n sound.load
        if self.noise_injector:
            add_noise = np.random.binomial(1, self.aug_conf.noise_prob)
            if add_noise:
                y = self.noise_injector.inject_noise(y)

        
        ##get ten file de ve hinh
        # nanlist=audio_path.split("/")
        # nanLs = nanlist[len(nanlist)-1]
        # name = nanLs.split(".")[0]+"_"+nanLs.split(".")[1]

        #t??n hi???u th??
        # fig1,ax22= plt.subplots()
        # plt.title('T??n hi???u th?? c???a c??u n??i \'anh c?? th??? g???i cho t??i kh??ng\'')
        # plt.plot(y)
        # plt.xlabel('Sample')
        # plt.ylabel('Amplitude')
        # fig1.savefig('/work/Source/deepspeech.pytorch/deepspeech_pytorch/quyenImg/'+name+'tinhieutho'+'.png')

        n_fft = int(self.sample_rate * self.window_size)#320
        win_length = n_fft#320
        hop_length = int(self.sample_rate * self.window_stride)#160
        # STFT
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=self.window)#array([[ 2.42148260e-01+0.00000000e+00j, -1.00020550e-01+0.00000000e+00j,

        spect, phase = librosa.magphase(D)# S = log(S+1)##array([[ 2.42148260e-01+0.00000000e+00j, -1.00020550e-01+0.00000000e+00j,



        # print("++**",audio_path)
        # fig, ax = plt.subplots()
        # img = librosa.display.specshow(librosa.amplitude_to_db(spect,ref=np.max), y_axis='log', x_axis='time', ax=ax)
        # ax.set_title(audio_path)
        # fig.savefig('/work/Source/deepspeech.pytorch/deepspeech_pytorch/quyenImg/'+name+'.png')

#c?? thanh ????? l???n bi??n ?????
        # log_spectrogram = librosa.amplitude_to_db(spect)
        # plt.figure(figsize=(12,8))
        # librosa.display.specshow(log_spectrogram, sr=self.sample_rate,
        # y_axis='log', x_axis='time',hop_length=160)
        # plt.xlabel("Time")
        # plt.ylabel("Frequency")
        # plt.colorbar(format="%+2.0f dB")
        # plt.title("Spectrogram (dB)")
        # plt.savefig('/work/Source/deepspeech.pytorch/deepspeech_pytorch/quyenImg/'+name+'.png')

        spect = np.log1p(spect)#ln(spect) tensor([[2.1684e-01, 9.5329e-02, 1.0469e-01,  ..., 1.2308e-03, 2.3625e-03,
        
        # fig2, ax2 = plt.subplots()
        # img2 = librosa.display.specshow(librosa.amplitude_to_db(spect,ref=np.max), y_axis='log', x_axis='time', ax=ax)
        # ax2.set_title(audio_path+"(log)")
        # fig2.savefig('/work/Source/deepspeech.pytorch/deepspeech_pytorch/quyenImg/'+name+"(log)"+'.png')

        spect = torch.FloatTensor(spect)#tensor([[2.1684e-01, 9.5329e-02, 1.0469e-01,  ..., 1.2308e-03, 2.3625e-03,
        if self.normalize:
            
            mean = spect.mean()#t??nh trung b??nh c???ng
            #mean=np.log1p(mean)

            std = spect.std()
            #std=np.log1p(std)#????? l???ch chu???n

            if (mean==torch.tensor(0) or std ==torch.tensor(0)):
                print("nan nan")
            
            spect.add_(-mean)
            spect.div_(std)

        if self.aug_conf and self.aug_conf.spec_augment:
            spect = spec_augment(spect)
        
        return spect

    def parse_transcript(self, transcript_path):
        raise NotImplementedError


# train_dataset = SpectrogramDataset(audio_conf=model.audio_conf,
#                                     manifest_filepath=to_absolute_path(cfg.data.train_manifest),
#                                     labels=model.labels,
#                                     normalize=True,
#                                     augmentation_conf=cfg.data.augmentation)

#Tr??? v??? x??? l?? spect v?? trainscript c???a t???ng  row trong csv. 
#Nh?? v?? d??? th?? train_dataset c?? 26k d??ng (FPT_VIVO), m???i d??ng l?? 1 c???p x??? l?? (spect, trainscript)
#SpectrogramDataset l??m ?????u v??o cho AudioDataLoader ????? x??? l?? collate_fn 

#trong class n??y c?? x??? l?? STFT, normolize
class SpectrogramDataset(Dataset, SpectrogramParser):
    def __init__(self,
                 audio_conf: SpectConfig,
                 manifest_filepath: str,
                 labels: list,
                 normalize: bool = False,
                 augmentation_conf: AugmentationConfig = None):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:

        /path/to/audio.wav,/path/to/audio.txt
        ...

        :param audio_conf: Config containing the sample rate, window and the window length/stride in seconds
        :param manifest_filepath: Path to manifest csv as describe above
        :param labels: List containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param augmentation_conf(Optional): Config containing the augmentation parameters
        """
        with open(manifest_filepath) as f:
            ids = f.readlines()
        ids = [x.strip().split(',') for x in ids]
        self.ids = ids
        self.size = len(ids)#t???ng d??ng trong file csv c???a train
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        super(SpectrogramDataset, self).__init__(audio_conf, normalize, augmentation_conf)

    def __getitem__(self, index):#v??o ????y v?? l???y c??i csv nguy??n th???y ????? chuy???n sang d???ng dataset g???m c??c samples, 1 samples c?? 32 m???u
        sample = self.ids[index]#index: m???u th??? m???y trong csv (d??ng m???y)#['/dataset/wav/FPTOpenSpeechData_Set001_V0.1_011816.wav', '/dataset/txt/FPTOpenSpeechData_Set001_V0.1_011816.txt']
        audio_path, transcript_path = sample[0], sample[1]
        spect = self.parse_audio(audio_path)#c?? x??? l?? STFT, normolize, #t???o ph??? cho ?????u v??o ??m thanh
        transcript = self.parse_transcript(transcript_path)#m???ng c??u tham chi???u ??? d???ng m?? k?? t??? trong lables #[59, 5, 92, 20, 48, 87, 92, 2, 58, 49, 54, 38, 92, 21, ...]
        return spect, transcript

# cho ???????ng d???n txt n?? ?????c n???i dung
    def parse_transcript(self, transcript_path):
        with open(transcript_path, 'r', encoding='utf8') as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
        transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        return transcript

    def __len__(self):
        return self.size

#h??m ri??ng t??nh tham chi???u ???????c g???i trong class AudioDataLoader
#the collate_fn argument is used to collate(?????i chi???u) lists of samples into batches.

#batch_sampler c?? 835 indicas, m???i indice c?? 32 m???u l?? 1 d??ng c???a dataset
# for indices in batch_sampler:
#     batch=[dataset[i] for i in indices];//32 d??ng
#     yield collate_fn(batch)
def _collate_fn(batch):#batch = [32 m???u, m???i m???u l?? (ma tr???n ph??? tensor 2 chi???u, transcript (??? d???ng m?? trong lables))]
    def func(p):
        return p[0].size(1)
#sample l?? 1 shape (spect, ...) => sample[0] l?? spect ph??? c???a c??u n??i, sample[1] l?? array c??c ascii c???a c??u n??i g???c
#spect n??y l?? ma tr???n 2 chi???u c??c s??? th???c (161, ...)=> sample[0].size(0)-> l???y s??? h??ng (do ki???u tensor n??n kh??c)
#batch ???????c s???p x???p theo gi???m d???n theo ????? d??i c??u n??i c???a ma tr???n ph???
    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)# s???p x???p batch theo sample[0].size(1)v???i sample[0] l?? ma tr???n ph??? c???a m???u th??? i v?? sample[0].size l?? k??ch th?????c ma tr???n 2 chi???u spect .size(1) l?? l???y c???t c???a spect, ????? d??i c??u, s??? ph???n t??? c???a c???t c???a ma tr???n 2  chi???u spect
    longest_sample = max(batch, key=func)[0]#l???y ma tr???n ??m thanh m?? c?? spect c?? ma tr???n nh??u c???t nh???t (c??u n??i d??i nh???t)
    freq_size = longest_sample.size(0)#l???y t???n s??? c???a file ??m thanh c?? nh??u c???t nh???t ->161
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)#l???y ????? d??i c??u n??i c???a file ??m thanh c?? nh??u c???t nh???t ->380 (b?????c th???i gian d??i nh???t)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)#ma tr???n 4 chi???u, input[x][0] l?? spect th??? x trong batchsize
    input_percentages = torch.FloatTensor(minibatch_size)#ma tr???n ch???a t??? l??? ????? d??i t???ng c??u n??i / c??u n??i d??i nahats
    target_sizes = torch.IntTensor(minibatch_size)# l?? m???ng k??ch th?????c transcript c???a 32  c??u g???c [141, 55, ..] ,=>m???u 1 c?? 141 k?? t???, m???u 2 c?? 55 k?? t??? ban ?????u random
    targets = []#m???ng 1 chi???u c??c k?? t??? c???a c??? 32 transcript li??n t???c . len(targets) l?? t???t c??? c??c k?? t??? c?? trong to??n b??? 32 transcript, ????? d??i 1898778 k?? t???
    for x in range(minibatch_size):#l???y n???i dung t???ng m???u b??? v??o ma tr???n inputs 4 chi???u ???? kh???i t???o
        sample = batch[x]#(tensor([[-0.4120, -0.4120, -0.4120,  ..., -0.3768, -0.4116, -0.3984]....), [13, 11, 49, 92, 16, 50, 2, 92, 2, ...])
        tensor = sample[0]#tensor([[0.4504, -0.987..]] l?? c???t [0] trong sample
        target = sample[1]#ascii [9,45,93,32,..]
        seq_length = tensor.size(1)#seq_length l?? ????? d??i c??u n??i ????n v??? time_step
        #print("###", tensor.size())
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)#inputs[x][0] l?? ma tr???n ph??? 2 chi???u c???a m???u x, copy gi?? tr??? tensor (?????c trong csv c???a d??ng x) qua; thu h???p ma tr???n theo chi???u 1 (4 chi???u 0 1 2 3), b???t ?????u ??? 0 v?? gi??? l???i seq_length c???t
        input_percentages[x] = seq_length / float(max_seqlength)# l???y s??? c???t c???a spect hi???n t???i / s??? c???t max-> ph???n tr??m
        #print("$$$$$$$$$$$$",input_percentages[x])
        target_sizes[x] = len(target)#s??? k?? t??? trong c??u n??i "c??ch ????? ??i"->10
        targets.extend(target)# add 2 ki???u d??? li???u kh??c nhau v??o list, vd 'aaa' v?? 1,  2
    targets = torch.IntTensor(targets)
    #inputs l?? m???ng 4 chi???u t???ng sample c???a 32 sample. 1 sample c?? tensor v?? ascii
    return inputs, targets, input_percentages, target_sizes

#  train_loader = AudioDataLoader(dataset=train_dataset,
#                                    num_workers=cfg.data.num_workers,
#                                    batch_sampler=train_sampler), ???????c g???i trong main

#Link: https://pytorch.org/docs/stable/data.html
# DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
#            batch_sampler=None, num_workers=0, collate_fn=None,
#            pin_memory=False, drop_last=False, timeout=0,
#            worker_init_fn=None, *, prefetch_factor=2, 
#            persistent_workers=False)

#help
    # map-style and iterable-style datasets,
    # customizing data loading order,
    # automatic batching,
    # single- and multi-process data loading,
    # automatic memory pinning.

#tham s??? dataset: ch??? ngu???n ????? load data t??? ????, when accessed with dataset[idx], could read the idx-th image (ki???u map-style)
#tham s??? : shuffle l???y m???u tu???n t??? hay x??o tr???n
#tham s???: sampler: ch??? ?????nh c??? th??? samples n??o b??? x??o tr???n. samples l?? 1 t???p g???m n(batch_size) m???u trong dataset. vd 32 m???u
#tham s???: batch_sampler l?? 32 m???u
#AudioLoader tr??? ra cu???i c??ng l?? m???ng c?? 835 ph???n t???, c??ch ????? chuy???n l?? d??ng h??m collate_fn****, v?? khi n??o d??ng 1 batch_size m???i yeild t???i x??? l?? c??c m???u ????? chuy???n
class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn
        #h??m ri??ng t??nh tham chi???u ???????c g???i trong class AudioDataLoader
        #the collate_fn argument is used to collate(?????i chi???u) lists of samples into batches.


class DSRandomSampler(Sampler):#ch???n 1 sample b???t k?? trong 865 batch
    """
    Implementation of a Random Sampler for sampling the dataset.
    Added to ensure we reset the start index when an epoch is finished.
    This is essential since we support saving/loading state during an epoch.
    """

    def __init__(self, dataset, batch_size=1, start_index=0):
        super().__init__(data_source=dataset)

        self.dataset = dataset
        self.start_index = start_index
        self.batch_size = batch_size
        ids = list(range(len(self.dataset)))#m???ng [0,...n] v???i n l?? d??ng trong train_csv
        self.bins = [ids[i:i + self.batch_size] for i in range(0, len(ids), self.batch_size)]

    def __iter__(self):#l???y ph???n t??? ti???p theo next iter(DSRandomSampler)
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = (
            torch.randperm(len(self.bins) - self.start_index, generator=g)
                .add(self.start_index)
                .tolist()
        )
        for x in indices:
            batch_ids = self.bins[x]
            np.random.shuffle(batch_ids)#x??o tr???n c??c n???i file ??m thanh trong minibatch
            yield batch_ids

    def __len__(self):
        return len(self.bins) - self.start_index

    def set_epoch(self, epoch):
        self.epoch = epoch

    def reset_training_step(self, training_step):
        self.start_index = training_step


class DSElasticDistributedSampler(DistributedSampler):
    """
    Overrides the ElasticDistributedSampler to ensure we reset the start index when an epoch is finished.
    This is essential since we support saving/loading state during an epoch.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, start_index=0, batch_size=1):
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank)
        self.start_index = start_index
        self.batch_size = batch_size
        ids = list(range(len(dataset)))
        self.bins = [ids[i:i + self.batch_size] for i in range(0, len(ids), self.batch_size)]
        self.num_samples = int(
            math.ceil(float(len(self.bins) - self.start_index) / self.num_replicas)
        )
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = (
            torch.randperm(len(self.bins) - self.start_index, generator=g)
                .add(self.start_index)
                .tolist()
        )

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank: self.total_size: self.num_replicas]
        assert len(indices) == self.num_samples
        for x in indices:
            batch_ids = self.bins[x]
            np.random.shuffle(batch_ids)
            yield batch_ids

    def __len__(self):
        return self.num_samples

    def reset_training_step(self, training_step):
        self.start_index = training_step
        self.num_samples = int(
            math.ceil(float(len(self.bins) - self.start_index) / self.num_replicas)
        )
        self.total_size = self.num_samples * self.num_replicas


def audio_with_sox(path, sample_rate, start_time, end_time):
    """
    crop and resample the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as tar_file:
        tar_filename = tar_file.name
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} trim {} ={} >/dev/null 2>&1".format(path, sample_rate,
                                                                                               tar_filename, start_time,
                                                                                               end_time)
        os.system(sox_params)
        y = load_audio(tar_filename)
        return y


def augment_audio_with_sox(path, sample_rate, tempo, gain):
    """
    Changes tempo and gain of the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as augmented_file:
        augmented_filename = augmented_file.name
        sox_augment_params = ["tempo", "{:.3f}".format(tempo), "gain", "{:.3f}".format(gain)]
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} {} >/dev/null 2>&1".format(path, sample_rate,
                                                                                      augmented_filename,
                                                                                      " ".join(sox_augment_params))
        os.system(sox_params)
        y = load_audio(augmented_filename)
        return y


def load_randomly_augmented_audio(path, sample_rate=16000, tempo_range=(0.85, 1.15),
                                  gain_range=(-6, 8)):
    """
    Picks tempo and gain uniformly, applies it to the utterance by using sox utility.
    Returns the augmented utterance.
    """
    low_tempo, high_tempo = tempo_range
    tempo_value = np.random.uniform(low=low_tempo, high=high_tempo)
    low_gain, high_gain = gain_range
    gain_value = np.random.uniform(low=low_gain, high=high_gain)
    audio = augment_audio_with_sox(path=path, sample_rate=sample_rate,
                                   tempo=tempo_value, gain=gain_value)
    return audio
