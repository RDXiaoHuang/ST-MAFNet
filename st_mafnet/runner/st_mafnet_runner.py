import torch

from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.metrics import masked_mae, masked_rmse, masked_mape
from basicts.data import SCALER_REGISTRY


def masked_mape_percent(prediction, target, null_val=0.0):
    return masked_mape(prediction, target, null_val) * 100


class STMAFNetRunner(SimpleTimeSeriesForecastingRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.metrics = cfg.get("METRICS", {"MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape_percent})
        self.forward_features = cfg["MODEL"].get("FORWARD_FEATURES", None)
        self.target_features = cfg["MODEL"].get("TARGET_FEATURES", None)

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        data = data[:, :, :, self.target_features]
        return data

    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
        future_data, history_data = data
        history_data = self.to_running_device(history_data)
        future_data = self.to_running_device(future_data)

        history_data = self.select_input_features(history_data)

        prediction = self.model(
            history_data=history_data,
            future_data=None,
            batch_seen=iter_num,
            epoch=epoch,
            train=train
        )

        batch_size, length, num_nodes, _ = future_data.shape
        assert list(prediction.shape)[:3] == [batch_size, length, num_nodes], \
            "error shape of the output, edit the forward function to reshape it to [B, L, N, C]"

        prediction = self.select_target_features(prediction)
        real_value = self.select_target_features(future_data)

        return prediction, real_value

    def train_iters(self, epoch: int, iter_index: int, data: tuple) -> torch.Tensor:
        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        prediction, real_value = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)

        prediction_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(prediction, **self.scaler["args"])
        real_value_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(real_value, **self.scaler["args"])

        if self.cl_param:
            cl_length = self.curriculum_learning(epoch=epoch)
            prediction_rescaled = prediction_rescaled[:, :cl_length, :, :]
            real_value_rescaled = real_value_rescaled[:, :cl_length, :, :]

        loss = self.metric_forward(self.loss, [prediction_rescaled, real_value_rescaled])

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, [prediction_rescaled, real_value_rescaled])
            self.update_epoch_meter("train_" + metric_name, metric_item.item())

        return loss

    @torch.no_grad()
    def val_iters(self, iter_index: int, data: tuple):
        prediction, real_value = self.forward(data=data, epoch=None, iter_num=iter_index, train=False)

        prediction_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(prediction, **self.scaler["args"])
        real_value_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(real_value, **self.scaler["args"])

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, [prediction_rescaled, real_value_rescaled])
            self.update_epoch_meter("val_" + metric_name, metric_item.item())

    @torch.no_grad()
    def test_iters(self, iter_index: int, data: tuple):
        prediction, real_value = self.forward(data=data, epoch=None, iter_num=iter_index, train=False)
        return prediction, real_value
