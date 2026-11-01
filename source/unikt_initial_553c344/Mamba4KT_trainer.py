"""Mamba4KT 模型训练器"""

from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import TRAINERS, get_logger
from utils.training import BaseTrainer

logger = get_logger(__name__)


@register_model_params("Mamba4KT")
class Mamba4KTModelParams(BaseParamConfig):
    def define_params(self) -> tuple[str, dict]:
        group_name = "Mamba4KT Parameters"
        params = {
            "d_model": {
                "type": int,
                "default": 128,
                "help": "Hidden dimension of the model (paper: {64,128,256})",
            },
            "n_blocks": {
                "type": int,
                "default": 5,
                "help": "Number of Mamba blocks (paper N=5)",
            },
            "d_state": {
                "type": int,
                "default": 16,
                "help": "SSM latent state dimension",
            },
            "d_conv": {
                "type": int,
                "default": 4,
                "help": "Conv1D kernel width in Mamba block",
            },
            "expand": {
                "type": int,
                "default": 2,
                "help": "Mamba internal expansion factor (Conv1D out channels = expand*d_model)",
            },
            "dropout": {
                "type": float,
                "default": 0.1,
                "help": "Dropout probability",
            },
            "l2": {
                "type": float,
                "default": 1e-5,
                "help": "L2 regularization coefficient for Rasch difficulty parameter (lambda in Eq.11)",
            },
            "epochs": {
                "type": int,
                "default": 150,
                "short": "ep",
                "help": "Number of training epochs",
            },
            "learning_rate": {
                "type": float,
                "default": 1e-3,
                "short": "lr",
                "help": "Learning rate (paper: {0.003,0.002,0.001,0.0001})",
            },
            "weight_decay": {
                "type": float,
                "default": 0.0,
                "short": "wd",
                "help": "Weight decay for optimizer",
            },
            "batch_size": {
                "type": int,
                "default": 64,
                "short": "bs",
                "help": "Batch size (paper=64)",
            },
        }
        return group_name, params


@TRAINERS.register("Mamba4KT")
class Mamba4KTTrainer(BaseTrainer):
    """Mamba4KT 模型训练器。"""

    def __init__(
        self, args: Any = None, data_src: Any = None, exp_manager: Any = None
    ) -> None:
        from model.Mamba4KT.Mamba4KT_data import Mamba4KTModelData

        model_data = Mamba4KTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(args)

        from model.Mamba4KT.Mamba4KT_model import Mamba4KT

        logger.info("Initializing Mamba4KT model...")
        metadata = data_src.get_metadata()
        n_pid = metadata.get("num_questions", 0)

        if n_pid > 0:
            logger.info(f"Mamba4KT: Using Rasch embeddings with {n_pid} questions")
        else:
            logger.warning("Mamba4KT: Problem ID not available, using skill-only model")

        model = Mamba4KT(
            num_c=metadata["num_skills"],
            n_pid=n_pid,
            d_model=args.d_model,
            n_blocks=args.n_blocks,
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
            dropout=args.dropout,
            l2=args.l2,
        )

        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

        super().__init__(model)

        early_stopping_cfg = None
        es_patience = getattr(args, "es_patience", None)
        if es_patience is not None:
            early_stopping_cfg = EarlyStoppingConfig(
                monitor=getattr(args, "es_monitor", "auc"),
                mode=getattr(args, "es_mode", "max"),
                patience=es_patience,
                min_delta=getattr(args, "es_min_delta", 0.0),
            )

        self.with_training(
            epochs=args.epochs,
            seed=args.seed,
            device=args.device,
            checkpoint_path=args.checkpoint_path,
        ).with_data(
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
            batch_size=args.batch_size,
        ).with_optimization(
            optimizer=optimizer,
            loss_fn=loss_fn,
            early_stopping=early_stopping_cfg,
        ).with_experiment(
            exp_manager=exp_manager,
            hyperparams=args,
            model_name="Mamba4KT",
            dataset_name=getattr(args, "dataset", ""),
        ).build()

    def _build_pid_data(
        self, question: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """构建 Rasch pid 数据，0 保留给填充位置。"""
        pid_data = question.clone() + 1
        return pid_data.masked_fill(~valid_mask.bool(), 0)

    def forward_pass(self, batch_data: tuple) -> dict[str, torch.Tensor]:
        """Mamba4KT 前向传播（next-item 约定）。

        out[t] 利用历史 0..t 预测 response[t+1]，待预测题目 q_{t+1} 已在模型内部左移注入。
        """
        sequence, response, mask, question = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        question = self._move_tensor_to_device(question)

        use_pid = self.model.n_pid > 0
        pid_data = self._build_pid_data(question, mask) if use_pid else None

        y_hat_full, c_reg_loss = self.model(
            sequence, response, mask, pid_data
        )  # [B, S]

        # next-item 对齐
        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full, response, mask, same_position=False
        )

        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)
        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        result = {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": y_hat,
        }
        if use_pid:
            result["c_reg_loss"] = c_reg_loss
        return result

    def test_forward_pass(self, batch_data):
        """测试前向传播，支持 windowlateauc_mean 评估。

        batch_data: (sequence, response, mask, late_group_id, true_labels, question)
        """
        sequence, response, mask, late_group_id, true_labels, question = batch_data

        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)
        question = self._move_tensor_to_device(question)

        use_pid = self.model.n_pid > 0
        valid_mask = late_group_id >= 0
        pid_data = self._build_pid_data(question, valid_mask) if use_pid else None

        y_hat_full, _ = self.model(sequence, response, mask, pid_data)  # [B, S]

        # windowlate 约定：每个窗口仅在最后一位（target_pos=p）mask=1，目标 response 被置 0 防泄漏。
        # next-item 下对目标位 p 的预测位于 y_hat_full[p-1]（使用历史 0..p-1 与题目 q_p），
        # 故用 mask[:, 1:] 选出目标位，与其前一位的预测对齐。
        target_mask = mask[:, 1:]
        y_hat = torch.masked_select(y_hat_full[:, :-1], target_mask)
        y_label = torch.masked_select(true_labels[:, 1:].float(), target_mask)
        group_ids = torch.masked_select(late_group_id[:, 1:], target_mask)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": self._generate_binary_predictions(y_hat, threshold=0.5),
            "y_score": y_hat,
            "y_prob": y_hat,
            "group_id": group_ids,
        }

    def _compute_loss(self, outputs: dict) -> torch.Tensor:
        """计算损失，包含 BCE 损失与 Rasch 正则化损失（论文 Eq. 11）。"""
        y_hat = outputs["y_hat"]
        y_label = outputs["y_label"]
        bce_loss = self.loss(y_hat, y_label)

        if "c_reg_loss" in outputs and self.model.n_pid > 0:
            return bce_loss + outputs["c_reg_loss"]
        return bce_loss
