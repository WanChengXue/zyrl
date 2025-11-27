import os
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# 设置torch默认数据类型为float32
torch.set_default_dtype(torch.float32)


class SLTrainer:
    def __init__(self, *, learning_config: dict):
        self._learning_config = learning_config
        self._epoch = learning_config.get("epoch", 100)
        self._batch_size = learning_config.get("batch_size", 128)
        self._learning_rate = learning_config.get("learning_rate", 0.001)
        self._weight_decay = learning_config.get("weight_decay", 0.0001)
        self._loss_threshold = learning_config.get("loss_threshold", 0.001)
        self._checkpoint_folder = learning_config.get("checkpoint_folder", None)
        self._device = learning_config.get("device", "cpu")
        self._batch_loss_log = []
        self._epoch_loss_log = []
        self._create_ckpt_folder()
        self._create_model_optimizer_and_scheduler()

    def _create_ckpt_folder(self):
        if not os.path.exists(self._checkpoint_folder):
            os.makedirs(self._checkpoint_folder)
        self._sl_saved_model_path = f"{self._checkpoint_folder}/model.pth"

    def _create_model_optimizer_and_scheduler(self):
        self._model = self._learning_config["model"]
        self._model.to(
            device=self._device, dtype=torch.float32
        )  # 确保模型参数为float32
        self._optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self._learning_rate,
            weight_decay=self._weight_decay,
        )
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer, step_size=10, gamma=0.1
        )
        self._mse_loss = nn.MSELoss()
        torch.save(self._model.state_dict(), self._sl_saved_model_path)

    def _create_dataloader(self, trained_data, label, mask=None):
        if mask is not None:
            dataset = TensorDataset(
                torch.tensor(trained_data, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32),
                torch.tensor(mask, dtype=torch.float32),
            )
        else:
            dataset = TensorDataset(
                torch.tensor(trained_data, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32),
            )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self._batch_size,
            shuffle=True,
        )
        return dataloader

    def learn(self):
        trained_data = self._learning_config["trained_data"]
        label = self._learning_config["label"]
        if "mask" in self._learning_config:
            mask = self._learning_config["mask"]
            self._mse_loss = nn.MSELoss(reduction="none")
        else:
            mask = None
        data_loader = self._create_dataloader(trained_data, label, mask)
        for epoch in tqdm(range(self._epoch)):
            epoch_loss = 0
            if mask is not None:
                for data, label, mask in data_loader:
                    predict_value = self._model(data.to(self._device))[0]
                    loss = self._mse_loss(predict_value, label.to(self._device))
                    loss[mask.bool().to(self._device) == False] = 0
                    loss = torch.sum(loss) / torch.sum(mask)
                    self._optimizer.zero_grad()
                    loss.backward()
                    self._optimizer.step()
                    self._scheduler.step()
                    self._batch_loss_log.append(loss.item())
                    epoch_loss += loss.item()
                    if loss.item() < self._loss_threshold:
                        break
            else:
                for data, label in data_loader:
                    predict_value = self._model(data.to(self._device))
                    loss = self._mse_loss(predict_value, label.to(self._device))
                    self._optimizer.zero_grad()
                    loss.backward()
                    self._optimizer.step()
                    self._scheduler.step()
                    self._batch_loss_log.append(loss.item())
                    epoch_loss += loss.item()
                    print(f"Epoch {epoch}, Loss: {loss.item()}")
                    if loss.item() < self._loss_threshold:
                        break
            self._epoch_loss_log.append(epoch_loss)
        torch.save(self._model.state_dict(), self._sl_saved_model_path)
        self.save_loss_log()

    def save_loss_log(self):
        plt.figure(figsize=(20, 15))
        plt.subplot(2, 1, 1)
        plt.plot(self._batch_loss_log)
        plt.title("Batch Loss Log")
        plt.subplot(2, 1, 2)
        plt.plot(self._epoch_loss_log)
        plt.title("Epoch Loss Log")
        plt.savefig(f"{self._checkpoint_folder}/sl_loss_log.png")
        plt.close()

    def get_checkpoint_path(self):
        return self._sl_saved_model_path
