import json
import os
import random
import time

import numpy as np
import torch.distributed as dist
import torch.utils.data.distributed
from apex import amp
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from warpctc_pytorch import CTCLoss

from deepspeech_pytorch.checkpoint import FileCheckpointHandler, GCSCheckpointHandler
from deepspeech_pytorch.configs.train_config import SGDConfig, AdamConfig, BiDirectionalConfig, UniDirectionalConfig, \
    FileCheckpointConfig, GCSCheckpointConfig
from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.loader.data_loader import SpectrogramDataset, DSRandomSampler, DSElasticDistributedSampler, \
    AudioDataLoader
from deepspeech_pytorch.logger import VisdomLogger, TensorBoardLogger
from deepspeech_pytorch.model import DeepSpeech, supported_rnns
from deepspeech_pytorch.state import TrainingState
from deepspeech_pytorch.testing import run_evaluation
from deepspeech_pytorch.utils import check_loss
from deepspeech_pytorch.mylogger import Logger
from deepspeech_pytorch.sendReport import sendReport
mylogg2er = Logger("/root/trainlog/mylog")
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(cfg):
    # Set seeds for determinism
    torch.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed_all(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    random.seed(cfg.training.seed)

    main_proc = True
    device = torch.device("cpu" if cfg.training.no_cuda else "cuda")

    is_distributed = os.environ.get("LOCAL_RANK")  # If local rank exists, distributed env

    if is_distributed:
        # when using NCCL, on failures, surviving nodes will deadlock on NCCL ops
        # because NCCL uses a spin-lock on the device. Set this env var and
        # to enable a watchdog thread that will destroy stale NCCL communicators
        os.environ["NCCL_BLOCKING_WAIT"] = "1"

        device_id = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(device_id)
        print(f"Setting CUDA Device to {device_id}")

        dist.init_process_group(backend=cfg.training.dist_backend.value)
        main_proc = device_id == 0  # Main process handles saving of models and reporting

    if OmegaConf.get_type(cfg.checkpointing) == FileCheckpointConfig:
        checkpoint_handler = FileCheckpointHandler(cfg=cfg.checkpointing)
    elif OmegaConf.get_type(cfg.checkpointing) == GCSCheckpointConfig:
        checkpoint_handler = GCSCheckpointHandler(cfg=cfg.checkpointing)
    else:
        raise ValueError("Checkpoint Config has not been specified correctly.")

    if main_proc and cfg.visualization.visdom:
        visdom_logger = VisdomLogger(id=cfg.visualization.id,
                                     num_epochs=cfg.training.epochs)
    if main_proc and cfg.visualization.tensorboard:
        tensorboard_logger = TensorBoardLogger(id=cfg.visualization.id,
                                               log_dir=to_absolute_path(cfg.visualization.log_dir),
                                               log_params=cfg.visualization.log_params)

    if cfg.checkpointing.load_auto_checkpoint:
        latest_checkpoint = checkpoint_handler.find_latest_checkpoint()
        if latest_checkpoint:
            cfg.checkpointing.continue_from = latest_checkpoint

    if cfg.checkpointing.continue_from:  # Starting from previous model
        state = TrainingState.load_state(state_path=to_absolute_path(cfg.checkpointing.continue_from))
        model = state.model
        if cfg.training.finetune:
            state.init_finetune_states(cfg.training.epochs)

        if main_proc and cfg.visualization.visdom:  # Add previous scores to visdom graph
            visdom_logger.load_previous_values(state.epoch, state.results)
        if main_proc and cfg.visualization.tensorboard:  # Previous scores to tensorboard logs
            tensorboard_logger.load_previous_values(state.epoch, state.results)
    else:
        # Initialise new model training
        with open(to_absolute_path(cfg.data.labels_path)) as label_file:
            labels = json.load(label_file)

# #cấu hình của model trong file train_config.py dòng 51
# @dataclass
# class BiDirectionalConfig:
#     rnn_type: RNNType = RNNType.lstm  # Type of RNN to use in model
#     hidden_size: int = 1024  # Hidden size of RNN Layer
#     hidden_layers: int = 7  # Number of RNN layers
        if OmegaConf.get_type(cfg.model) is BiDirectionalConfig:
            model = DeepSpeech(rnn_hidden_size=cfg.model.hidden_size,
                               nb_layers=cfg.model.hidden_layers,
                               labels=labels,
                               rnn_type=supported_rnns[cfg.model.rnn_type.value],
                               audio_conf=cfg.data.spect,
                               bidirectional=True)
        elif OmegaConf.get_type(cfg.model) is UniDirectionalConfig:
            model = DeepSpeech(rnn_hidden_size=cfg.model.hidden_size,
                               nb_layers=cfg.model.hidden_layers,
                               labels=labels,
                               rnn_type=supported_rnns[cfg.model.rnn_type.value],
                               audio_conf=cfg.data.spect,
                               bidirectional=False,
                               context=cfg.model.lookahead_context)
        else:
            raise ValueError("Model Config has not been specified correctly.")

        state = TrainingState(model=model)
        state.init_results_tracking(epochs=cfg.training.epochs)

    # Data setup
    evaluation_decoder = GreedyDecoder(model.labels)  # Decoder used for validation
    train_dataset = SpectrogramDataset(audio_conf=model.audio_conf,
                                       manifest_filepath=to_absolute_path(cfg.data.train_manifest),
                                       labels=model.labels,
                                       normalize=True,
                                       augmentation_conf=cfg.data.augmentation)#cấu hình spect, ids=[[dòng 1],[dognf 2]..], lables_=dict 
    test_dataset = SpectrogramDataset(audio_conf=model.audio_conf,
                                      manifest_filepath=to_absolute_path(cfg.data.val_manifest),
                                      labels=model.labels,
                                      normalize=True)
    if not is_distributed:
        train_sampler = DSRandomSampler(dataset=train_dataset,
                                        batch_size=cfg.data.batch_size,
                                        start_index=state.training_step)#DSRandomSampler để  chọn 1 bộ minibatch bất kì và xáo trộn nội dung trong minibatch
    else:
        train_sampler = DSElasticDistributedSampler(dataset=train_dataset,
                                                    batch_size=cfg.data.batch_size,
                                                    start_index=state.training_step)
    train_loader = AudioDataLoader(dataset=train_dataset,
                                   num_workers=cfg.data.num_workers,
                                   batch_sampler=train_sampler)#AudioLoader có hàm collate_fn để xử lí 1 minibatch được chọn, trả ra cuối cùng là mảng có 835 phần tử(đối với FPT, VIVOS), mỗi phần tử của audio loader là 1 mảng gồm batch_size mẫu
    test_loader = AudioDataLoader(dataset=test_dataset,
                                  num_workers=cfg.data.num_workers,
                                  batch_size=cfg.data.batch_size)

    model = model.to(device)
    parameters = model.parameters()
    if OmegaConf.get_type(cfg.optim) is SGDConfig:#mặc định ở dòng 8 trong train_config
        optimizer = torch.optim.SGD(parameters,
                                    lr=cfg.optim.learning_rate,
                                    momentum=cfg.optim.momentum,
                                    nesterov=True,
                                    weight_decay=cfg.optim.weight_decay)
    elif OmegaConf.get_type(cfg.optim) is AdamConfig:
        optimizer = torch.optim.AdamW(parameters,
                                      lr=cfg.optim.learning_rate,
                                      betas=cfg.optim.betas,
                                      eps=cfg.optim.eps,
                                      weight_decay=cfg.optim.weight_decay)
    else:
        raise ValueError("Optimizer has not been specified correctly.")

    model, optimizer = amp.initialize(model, optimizer,
                                      enabled=not cfg.training.no_cuda,
                                      opt_level=cfg.apex.opt_level,
                                      loss_scale=cfg.apex.loss_scale)
    if state.optim_state is not None:
        optimizer.load_state_dict(state.optim_state)
    if state.amp_state is not None:
        amp.load_state_dict(state.amp_state)

    # Track states for optimizer/amp
    state.track_optim_state(optimizer)
    if not cfg.training.no_cuda:
        state.track_amp_state(amp)

    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[device_id])
    #print(model)
    print("Number of parameters: %d" % DeepSpeech.get_param_size(model))

    criterion = CTCLoss()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    for epoch in range(state.epoch, cfg.training.epochs):#1 epoch là duyệt hết cả csv của train
        model.train()
        end = time.time()
        start_epoch_time = time.time()
        state.set_epoch(epoch=epoch)
        train_sampler.set_epoch(epoch=epoch)
        train_sampler.reset_training_step(training_step=state.training_step)
        for i, (data) in enumerate(train_loader, start=state.training_step):#duyệt qua từng minibatch (gồm 32 mẫu)
            state.set_training_step(training_step=i)
            inputs, targets, input_percentages, target_sizes = data#inputs[x][0] chứ spect thứ x trong batchsixe, input_percenttages: tỉ lệ độ dài câu từng câu nói trong minibatch/độ dài max, target: array [[...mã ascii]]
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()#tensor([699, 682, 656, 560, 553, 517, 514, 502, 464, 458, 423, 412, 406, 349, ...] laayss tỉ lệ X độ dài max để ra độ dài thực sự từng câu nói
            # measure data loading time
            data_time.update(time.time() - end)
            inputs = inputs.to(device)
            #đưa inputs gồm 32 mẫu qua mô hình học sâu, và kích thước thực sự từng câu nói (độ dài bước thời gian phổ)
            out, output_sizes = model(inputs, input_sizes)#như out: 3 chiều, outputsize: 1 chiều : chứa kích thước mô hình dự đoán cho văn bản kq
            out = out.transpose(0, 1)  # TxNxH sau khi tranpose : out 3 chiều (bị đổi chiều 0 và 1)=> out (190x3x93)

            float_out = out.float()  # ensure float32 for loss
            loss = criterion(float_out, targets, output_sizes, target_sizes).to(device)
            loss = loss / inputs.size(0)  # average the loss by minibatch, tổng loss chia cho số spect trong batch đó 
            loss_value = loss.item()


            # Check to ensure valid loss was calculated
            valid_loss, error = check_loss(loss, loss_value)
            if valid_loss:
                optimizer.zero_grad()

                # compute gradient, SGD chuẩn hóa SGD cập nhật trọng số
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), cfg.optim.max_norm)
                optimizer.step()
            else:
                print(error)
                print('Skipping grad update')
                loss_value = 0

            state.avg_loss += loss_value
            losses.update(loss_value, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                (epoch + 1), (i + 1), len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses))
            if main_proc and cfg.checkpointing.checkpoint_per_iteration:
                checkpoint_handler.save_iter_checkpoint_model(epoch=epoch, i=i, state=state)
            del loss, out, float_out

        state.avg_loss /= len(train_dataset)

        epoch_time = time.time() - start_epoch_time
        print('Training Summary Epoch: [{0}]\t'
              'Time taken (s): {epoch_time:.0f}\t'
              'Average Loss {loss:.3f}\t'.format(epoch + 1, epoch_time=epoch_time, loss=state.avg_loss))
        mylogg2er.info('Training Summary Epoch: [{0}]\t'
              'Time taken (s): {epoch_time:.0f}\t'
              'Average Loss {loss:.3f}\n'.format(epoch + 1, epoch_time=epoch_time, loss=state.avg_loss))
        file_object = open('/root/epoch.log', 'a')
        file_object.write('Training Summary Epoch: [{0}]\t'
              'Time taken (s): {epoch_time:.0f}\t'
              'Average Loss {loss:.3f}\n'.format(epoch + 1, epoch_time=epoch_time, loss=state.avg_loss))
        file_object.close()
        with torch.no_grad():
            wer, cer, output_data, wer2, cer2 = run_evaluation(test_loader=test_loader,
                                                   device=device,
                                                   model=model,
                                                   decoder=evaluation_decoder,
                                                   target_decoder=evaluation_decoder)

        state.add_results(epoch=epoch,
                          loss_result=state.avg_loss,
                          wer_result=wer,
                          cer_result=cer)
 
        print('Validation Summary Epoch: [{0}]\t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(epoch + 1, wer=wer, cer=cer))
        # mylogg2er.info('Validation Summary Epoch: [{0}]\t'
        #       'Average WER {wer:.3f}\t'
        #       'Average CER {cer:.3f}\t'.format(epoch + 1, wer=wer, cer=cer))
        file_object = open('/root/epoch.log', 'a')
        file_object.write('Validation Summary Epoch: [{0}]\t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(epoch + 1, wer=wer, cer=cer))
        file_object.write('Validation Summary Epoch: [{0}]\t'
            'Average WER2 {wer:.3f}\t'
            'Average CER2 {cer:.3f}\n'.format(epoch + 1, wer=wer2, cer=cer2))
        file_object.close()
        if main_proc and cfg.visualization.visdom:
            visdom_logger.update(epoch, state.result_state)
        if main_proc and cfg.visualization.tensorboard:
            tensorboard_logger.update(epoch, state.result_state, model.named_parameters())

        if main_proc and cfg.checkpointing.checkpoint:  # Save epoch checkpoint
            checkpoint_handler.save_checkpoint_model(epoch=epoch, state=state)
        # anneal lr
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] / cfg.optim.learning_anneal
        print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))
        file_object = open('/root/epoch.log', 'a')
        file_object.write('Learning rate annealed to: {lr:.6f}\n'.format(lr=g['lr']))
        file_object.close()
        try:
            #  print('Training Summary Epoch: [{0}]\t'
            #   'Time taken (s): {epoch_time:.0f}\t'
            #   'Average Loss {loss:.3f}\t'.format(epoch + 1, epoch_time=epoch_time, loss=state.avg_loss))/////////
            note = "Đổi tham số train_config: type: rnn.gru epochs: int = 50, batch_size: int = 30, hidden_size: int = 1600, hidden_layers: int = 7, file train_manifest: vinfptunk_train.csv, vinfptunk_dev.csv", 
            sendReport(epoch+1,'{:.3f}'.format(epoch_time),'{:.3f}'.format(state.avg_loss), '{:.3f}'.format(wer),'{:.3f}'.format(cer),"{:.6f}".format(g['lr']), note)
        except Exception as esss:
            print('Error :', esss)
        if main_proc and (state.best_wer is None or state.best_wer > wer):
            checkpoint_handler.save_best_model(epoch=epoch, state=state)
            state.set_best_wer(wer)
            state.reset_avg_loss()
        state.reset_training_step()  # Reset training step for next epoch
