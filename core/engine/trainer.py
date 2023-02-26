from random import random
import time
import os
import warnings
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

from thop import profile
from warmup_scheduler import GradualWarmupScheduler

from ..dataset import FeatureDataset
from ..dataset import createPartition
from ..models import FeatureNet, FeatureNetLite, SCNN, MCNN, MsvNetLite, BaselineNet, BaselineNet2, VoxNet
from ..utils import setup_logger
from ..utils import LayerOutHook
from ..utils import plot_with_labels, plot3D_with_labels


class Trainer:
    def __init__(self, cfg):
        self.base_lr = cfg.base_lr
        self.num_train = cfg.num_train
        self.num_val_test = cfg.num_val_test
        self.arch = cfg.arch
        self.train_epochs = cfg.train_epochs
        self.resolution = cfg.resolution
        self.num_cuts = cfg.num_cuts  # obsolete
        if cfg.train_batchsize > cfg.num_train * cfg.num_of_class:
            # batch szie should be smaller than the number of samples
            cfg.train_batchsize = cfg.num_train * cfg.num_of_class
        self.train_bs = cfg.train_batchsize
        self.val_bs = cfg.val_batchsize
        self.weight_decay = cfg.weight_decay
        self.warmup_epochs = cfg.warmup_epochs
        self.lr_sch = cfg.lr_sch
        self.optim = cfg.optim
        self.data_aug = cfg.data_aug
        self.pretrained = cfg.pretrained
        self.simsiam_pretrained = cfg.simsiam_pretrained
        self.freeze = cfg.freeze
        self.val_interval = cfg.val_interval
        self.data_path = cfg.data_path
        self.num_of_class = cfg.num_of_class 
        self.workers = cfg.workers
        self.optimizer = None
        self.scheduler = None
        self.output_type = '3d'
        self.model_path = cfg.model_path
        self.shapetypes = ['O ring', 'Through hole', 'Blind hole', 'Triangular passage', 'Rectangular passage', 
                           'Circular through slot', 'Triangular through slot', 'Rectangular through slot', 
                           'Rectangular blind slot', 'Triangular pocket', 'Rectangular pocket', 'Circular end pocket',
                           'Triangular blind step', 'Circular blind step', 'Rectangular blind step', 'Rectangular through step', 
                           '2-sides through step', 'Slanted through step', 'Chamfer', 'Round', 'Vertical circular end blind slot', 
                           'Horizontal circular end blind slot', '6-sides passage', '6-sides pocket'
                           ]

        # training output directory
        time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        self.output_dir = os.path.join(cfg.output_dir, time_str)
        self.summary_writer = SummaryWriter(
            os.path.join(self.output_dir, 'SummaryWriter'))
        self.logger = setup_logger(os.path.join(
            self.output_dir, 'log.txt'), 'FeatureRecognition')

        # print and log config
        self.logger.info('Training config:')
        for s in str(cfg).split(','):
            self.logger.info(s)
            print(s)

        # set device
        if cfg.device == 'gpu':
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        elif cfg.device == "cpu":
            self.device = torch.device("cpu")

        # data augmentation ops
        augmentations = None
        if self.data_aug == True:
            if self.simsiam_pretrained is not None:
                augmentations = ['randomRotation']
            else:
                augmentations = ['randomRotation', 'randomScaleCrop']
            s = 'Data augmentation is enabled: '
            for op in augmentations:
                s += op + ' '
            self.logger.info(s)
            print(s)

        # define the network
        self.build_net()

        # create dataloaders
        partition = createPartition(self.data_path,
                                    self.num_of_class,
                                    self.resolution,
                                    self.num_train,
                                    self.num_val_test)

        training_set = FeatureDataset(partition['train'],
                                      resolution=self.resolution,
                                      output_type=self.output_type,
                                      num_cuts=self.num_cuts,
                                      data_augmentation=augmentations)
        self.trainloader = torch.utils.data.DataLoader(training_set,
                                                       batch_size=self.train_bs,
                                                       shuffle=True,
                                                       num_workers=self.workers,
                                                       pin_memory=True,
                                                       drop_last=False)

        val_set = FeatureDataset(partition['val'],
                                 resolution=self.resolution,
                                 output_type=self.output_type,
                                 num_cuts=self.num_cuts,
                                 data_augmentation=None)
        self.validloader = torch.utils.data.DataLoader(val_set,
                                                       batch_size=self.val_bs,
                                                       shuffle=False,
                                                       num_workers=self.workers,
                                                       pin_memory=True,
                                                       drop_last=False)

        test_set = FeatureDataset(partition['test'],
                                  resolution=self.resolution,
                                  output_type=self.output_type,
                                  num_cuts=self.num_cuts,
                                  data_augmentation=None)
        self.testloader = torch.utils.data.DataLoader(test_set,
                                                      batch_size=self.val_bs,
                                                      shuffle=False,
                                                      num_workers=self.workers,
                                                      pin_memory=True,
                                                      drop_last=False)

        # define the criterion and optimizer
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.build_optim()
        if self.lr_sch != 'constant':
            self.build_lr_sch()

    def build_net(self):
        if self.arch == 'FeatureNet':
            self.net = FeatureNet(num_classes=self.num_of_class)
            self.output_type = '3d'
        elif self.arch == 'FeatureNetLite':
            self.net = FeatureNetLite(
                scale=1.0, num_classes=self.num_of_class, dropout_prob=0., class_expand=1280)
            self.output_type = '3d'
        elif self.arch == 'MsvNet':
            scnn = SCNN(num_classes=self.num_of_class, pretraining=False)
            self.net = MCNN(
                model=scnn, num_classes=self.num_of_class, num_cuts=self.num_cuts)
            self.output_type = '2d_multiple'
        elif self.arch == 'MsvNetLite':
            self.net = MsvNetLite(scale=1.0, num_classes=self.num_of_class,
                                  dropout_prob=0., class_expand=1280, num_cuts=self.num_cuts)
            self.output_type = '2d_multiple'
        elif self.arch == 'BaselineNet':
            '''
                paper: Identifying manufacturability and machining processes using deep 3D convolutional networks
            '''
            self.net = BaselineNet(num_classes=self.num_of_class, input_shape=(
                self.resolution, self.resolution, self.resolution))
            self.output_type = '3d'
        elif self.arch == 'BaselineNet2':
            '''
                paper: Part machining feature recognition based on a deep learning method
            ''' 
            self.net = BaselineNet2(num_classes=self.num_of_class, input_shape=(
                self.resolution, self.resolution, self.resolution))
            self.output_type = '3d'
        elif self.arch == 'VoxNet':
            self.net = VoxNet(num_classes=self.num_of_class, input_shape=(
                self.resolution, self.resolution, self.resolution))
            self.output_type = '3d'
        else:
            raise ValueError('Invalid network type')
        # if running on GPU and we want to use cuda move model there
        self.net.to(self.device)

    def load_simsiam_pretrained_model(self):
        # freeze all layers but the last classifier layer
        if self.freeze:
            for name, param in self.net.named_parameters():
                if name not in ['classifier.weight', 'classifier.bias']:
                    param.requires_grad = False
        # init the classifier layer
        self.net.classifier.weight.data.normal_(mean=0.0, std=0.01)
        self.net.classifier.bias.data.zero_()

        if not self.freeze:
            classifier_parameters, model_parameters = [], []
            for name, param in self.net.named_parameters():
                if name in {'classifier.weight', 'classifier.bias'}:
                    classifier_parameters.append(param)
                else:
                    model_parameters.append(param)
            # set different learning rate for the classifier
            param_groups = [dict(params=classifier_parameters, lr=self.base_lr)]
            param_groups.append(dict(params=model_parameters, lr=0.001))
            #self.optimizer = optim.SGD(param_groups, 0, momentum=0.9, weight_decay=self.weight_decay)
            self.optimizer = optim.Adam(param_groups, 0, weight_decay=self.weight_decay)
            print(self.optimizer)
            self.build_lr_sch()

        # load from pre-trained, before DistributedDataParallel constructor
        if self.simsiam_pretrained:
            if os.path.isfile(self.simsiam_pretrained):
                print("=> loading checkpoint '{}'".format(
                    self.simsiam_pretrained))
                checkpoint = torch.load(
                    self.simsiam_pretrained, map_location="cpu")

                # rename moco pre-trained keys
                state_dict = checkpoint['state_dict']
                for k in list(state_dict.keys()):
                    # retain only encoder up to before the embedding layer
                    if k.startswith('encoder') and not k.startswith('encoder.classifier'):
                        # remove prefix
                        state_dict[k[len("encoder."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]

                # check whether pretrained parameters loading successfully
                msg = self.net.load_state_dict(state_dict, strict=False)
                assert set(msg.missing_keys) == {
                    "classifier.weight", "classifier.bias"}

                s = '=> pre-trained model loaded from {}'.format(
                    self.simsiam_pretrained)
                print(s)
                self.logger.info(s)
            else:
                s = "=> no checkpoint found at '{}'".format(
                    self.simsiam_pretrained)
                print(s)
                self.logger.info(s)

    def load_params(self, model_path):
        if os.path.isfile(model_path):
            s = "=> loading model params '{}'".format(model_path)
            print(s)
            self.logger.info(s)
            checkpoint = torch.load(model_path)
            try:
                state_dict = checkpoint['state_dict']
            except:
                state_dict = checkpoint
            msg = self.net.load_state_dict(state_dict, strict=False)
            print(msg)

    def build_lr_sch(self):
        # TODO scheduler parameters
        # multistepLR remember the real decay epoch = milestone + warmup epoch
        if self.lr_sch == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.1)
        elif self.lr_sch == 'multistep':
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[60, 80], gamma=0.1)
        elif self.lr_sch == 'exp':
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.99)
        elif self.lr_sch == 'cos':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.train_epochs, eta_min=0)
        elif self.lr_sch == 'constant':
            self.scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda epoch: self.base_lr)
        else:
            raise ValueError('Invalid lr_sch')

        if self.warmup_epochs > 0:
            temp_scheduler = self.scheduler
            self.scheduler = GradualWarmupScheduler(self.optimizer,
                                                    multiplier=1,
                                                    total_epoch=self.warmup_epochs,
                                                    after_scheduler=temp_scheduler)

    def build_optim(self):
        # TODO optimizer parameters
        if self.optim == 'adam':
            self.optimizer = optim.Adam(
                self.net.parameters(), lr=self.base_lr, weight_decay=self.weight_decay)
        elif self.optim == 'sgdm':
            self.optimizer = optim.SGD(
                self.net.parameters(), lr=self.base_lr, momentum=0.9, weight_decay=self.weight_decay)
        elif self.optim == 'rmsprop':
            self.optimizer = optim.RMSprop(
                self.net.parameters(), lr=self.base_lr, weight_decay=self.weight_decay)
        elif self.optim == 'adamw':
            self.optimizer = optim.AdamW(
                self.net.parameters(), lr=self.base_lr, weight_decay=self.weight_decay)
        else:
            raise ValueError('Invalid optimizer')

    def train(self):
        # load pretrained model
        if self.simsiam_pretrained is not None and self.pretrained is None:
            self.load_simsiam_pretrained_model()

        if self.pretrained is not None and self.simsiam_pretrained is None:
            self.load_params(self.pretrained)

        # input data shape
        if self.output_type == '3d':
            input_shape = (1, self.resolution,
                           self.resolution, self.resolution)
            input = torch.randn(1, 1, self.resolution,
                                self.resolution, self.resolution).to(self.device)
        elif self.output_type == '2d_multiple':
            input_shape = (self.num_cuts, 3, self.resolution, self.resolution)
            input = torch.randn(
                1, self.num_cuts, 3, self.resolution, self.resolution).to(self.device)

        # netowrl structure description
        summary(self.net, input_shape)

        # calc the network macs & params
        macs, params = profile(self.net, inputs=(input, ))
        macs_params_info = 'model size: MACS: {}G || Parameters: {}M'.format(
            macs / 1.e9, params / 1.e6)
        self.logger.info(macs_params_info)
        print(macs_params_info)

        print('\n surpvised training the network with labeled data ...')
        best_acc = 0.
        pbar = tqdm(range(self.train_epochs))
        for epoch in pbar:
            # train
            start_time = time.time()
            train_acc, train_loss = self.train_epoch()
            train_epoch_time = time.time()-start_time

            # validate
            if (epoch+1) % self.val_interval == 0:
                start_time = time.time()
                val_acc, val_loss = self.valtest_epoch(testval='val')
                val_epoch_time = time.time()-start_time

                # save best model
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save({'state_dict': self.net.state_dict()},
                               os.path.join(self.output_dir, 'best_model.pth'))
                    self.logger.info(
                        'Best model saved. at epoch {} with val_acc {:.2f}%'.format(epoch+1, val_acc))

                # update validate log
                self.summary_writer.add_scalar(
                    'train/val_acc', val_acc, epoch+1)
                self.summary_writer.add_scalar(
                    'train/val_loss', val_loss, epoch+1)
                self.logger.info('Eval at epoch {} with val_acc {:.2f}%, val_loss {:.5f} took {:.2f}s'.format(
                    epoch+1, val_acc, val_loss, val_epoch_time))

            cur_lr = self.optimizer.param_groups[0]['lr']
            self.summary_writer.add_scalar('train/lr', cur_lr, epoch+1)

            # update training log
            self.summary_writer.add_scalar(
                'train/train_acc', train_acc, epoch+1)
            self.summary_writer.add_scalar(
                'train/train_loss', train_loss, epoch+1)
            log_str = "epoch: {}/{} | Train Time: {:.2f}s | LR: {:.8f} | train_acc: {:.2f}%, train_loss: {:.5f}".format(
                epoch+1, self.train_epochs, train_epoch_time, cur_lr, train_acc, train_loss)
            pbar.set_description(log_str)
            self.logger.info(log_str)

    def train_epoch(self):
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        if self.scheduler is not None:
            self.scheduler.step()

        return 100.*correct/total, train_loss

    def valtest_epoch(self, testval='val'):
        if testval == 'val':
            loader = self.validloader
        elif testval == 'test':
            loader = self.testloader
        else:
            raise ValueError('Invalid testval loader')

        self.net.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)

                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return 100.*correct/total, val_loss

    def infer(self, inputs):
        self.net.eval()
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.net(inputs).max(1)[1].cpu().numpy()
        return outputs

    def estimate_infer_time(self):
        if self.output_type == '3d':
            dummy_input = torch.randn(1, 1, self.resolution,
                                self.resolution, self.resolution).to(self.device)
        elif self.output_type == '2d_multiple':
            dummy_input = torch.randn(
                1, self.num_cuts, 3, self.resolution, self.resolution).to(self.device)

        # warmup 20 times
        for _ in range(20):
            _ = self.infer(dummy_input)

        # estimate average inference time
        start_time = time.perf_counter()
        for _ in range(1000):
            _ = self.infer(dummy_input)
        end_time = time.perf_counter() - start_time

        # average =  end_time / 1000 (s)
        # ms = average * 1000 (ms)
        return end_time

    def tsne_visualize(self):
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        if self.model_path is not None:
            self.load_params(self.model_path)
        else:
            # load pretrained model
            if self.simsiam_pretrained is not None and self.pretrained is None:
                self.load_simsiam_pretrained_model()

            if self.pretrained is not None and self.simsiam_pretrained is None:
                self.load_params(self.pretrained)

        self.net.eval()
        layer_name = 'flatten'
        hook = LayerOutHook(self.net, layer_name)

        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

         # take n models per class for TSNE visualization
        num_data_vis = 40
        dim = 2
        partition = createPartition(self.data_path,
                                    self.num_of_class,
                                    self.resolution,
                                    self.num_train,
                                    num_val_test=num_data_vis)
        test_set = FeatureDataset(partition['test'],
                                  resolution=self.resolution,
                                  output_type=self.output_type,
                                  num_cuts=self.num_cuts,
                                  data_augmentation=None)
        testloader = torch.utils.data.DataLoader(test_set,
                                                 batch_size=num_data_vis*self.num_of_class,
                                                 shuffle=False,
                                                 num_workers=self.workers,
                                                 pin_memory=True,
                                                 drop_last=False)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)

                _ = self.net(inputs)
                last_layer = hook.output

                tsne = TSNE(perplexity=100, n_components=dim, init='pca', n_iter=5000)
                low_dim_embs = tsne.fit_transform(last_layer.cpu().data.numpy())
                labels = targets.cpu().numpy()
                if dim == 2:
                    plot_with_labels(low_dim_embs, labels)
                elif dim == 3:
                    plot3D_with_labels(low_dim_embs, labels)

        plt.ioff()

    def draw_ROC_CM(self):
        """
        Draw ROC curve
        """
        from sklearn.metrics import roc_curve, auc, confusion_matrix
        import matplotlib.pyplot as plt

        if self.model_path is not None:
            self.load_params(self.model_path)
        else:
            # load pretrained model
            if self.simsiam_pretrained is not None and self.pretrained is None:
                self.load_simsiam_pretrained_model()

            if self.pretrained is not None and self.simsiam_pretrained is None:
                self.load_params(self.pretrained)

        self.net.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            scores_list = []
            labels_list = []
            preds_list = []
            for batch_idx, (inputs, targets) in tqdm(enumerate(self.testloader)):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                scores = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)

                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                scores_list.append(scores.cpu().numpy())
                labels_list.append(targets.cpu().numpy())
                preds_list.extend(preds.cpu().numpy().tolist())
            scores = np.concatenate(scores_list)
            labels = np.concatenate(labels_list)

        print('ACC: ', 100.*correct/total)

        # Assuming you have true labels in labels and predicted scores in scores
        n_classes = len(self.shapetypes)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(labels == i, scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curve for each class
        plt.figure(figsize=(8, 6))
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label=f'ROC curve (AUC = {roc_auc[i]:.2f}) for class {self.shapetypes[i]}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multiclass ROC Curve')
        plt.legend(loc="lower right")
        plt.show()

        # Draw ROC curve
        cm = confusion_matrix(labels, preds_list)
        # Plot confusion matrix as an image
        plt.imshow(cm)
        plt.colorbar()
        plt.xticks(np.arange(len(self.shapetypes)), self.shapetypes, rotation=90)
        plt.yticks(np.arange(len(self.shapetypes)), self.shapetypes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()



def train_eval_model(cfg):
    warnings.filterwarnings('ignore', '.*output shape of zoom.*')

    trainer = Trainer(cfg)
    # train the model
    trainer.train()
    # load best model and evalute the model
    trainer.load_params(os.path.join(trainer.output_dir, 'best_model.pth'))
    val_acc, _ = trainer.valtest_epoch(testval='val')
    test_acc, _ = trainer.valtest_epoch(testval='test')
    # log results
    result_str = '\n\nVal Acc: {:.2f} | Test Acc: {:.2f}'.format(
        val_acc, test_acc)
    print(result_str)
    trainer.logger.info(result_str)


def eval_model(cfg):
    warnings.filterwarnings('ignore', '.*output shape of zoom.*')

    trainer = Trainer(cfg)
    # load best model and evalute the model
    if cfg.model_path is not None:
        trainer.load_params(cfg.model_path)
    else:
        return None
    val_acc, _ = trainer.valtest_epoch(testval='val')
    test_acc, _ = trainer.valtest_epoch(testval='test')
    # log results
    result_str = '\n\nVal Acc: {:.2f} | Test Acc: {:.2f}'.format(
        val_acc, test_acc)
    print(result_str)
    trainer.logger.info(result_str)


def infer_time_test(cfg):
    warnings.filterwarnings('ignore', '.*output shape of zoom.*')

    trainer = Trainer(cfg)
    avg_infer_time = trainer.estimate_infer_time()

    # log results
    result_str = '\n\n average inference time(on {:s}): {:.2f}ms'.format(cfg.device, avg_infer_time)
    print(result_str)
    trainer.logger.info(result_str)


def draw_TSNE(cfg):
    warnings.filterwarnings('ignore', '.*output shape of zoom.*')

    trainer = Trainer(cfg)
    trainer.tsne_visualize()

def draw_ROC_CM(cfg):
    warnings.filterwarnings('ignore', '.*output shape of zoom.*')

    cfg.val_batchsize = 256 # may be fast
    trainer = Trainer(cfg)
    trainer.draw_ROC_CM()