# https://github.com/Intelligent-Systems-Lab/FedML/blob/master/fedml_api/standalone/fedavg/my_model_trainer_classification.py

import logging
import torch
from torch import nn
from fedml_core.trainer.model_trainer import ModelTrainer


class ShaTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        # something magic here
        model = self.model

        model.to(device)
        model.train()

        if self.scheduler is not None:
            self.scheduler.step()
            args.lr = self.scheduler.get_lr()[0]
            # print(args.lr)

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)

        if self.scheduler is not None:
            self.scheduler.step()

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_size = args.batch_size
            lstm_state = model.zero_state(batch_size=batch_size, device=device)
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                optimizer.zero_grad()
                log_probs, lstm_state = model(x, lstm_state)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                self.id, epoch, sum(epoch_loss) / len(epoch_loss)))
    

    def test(self, test_data, device, args):
        # something magic here
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        batch_size = args.batch_size

        with torch.no_grad():
            lstm_state = model.zero_state(batch_size=batch_size, device=device)
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred, _ = model(x, lstm_state)
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        return metrics
        
    
    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
