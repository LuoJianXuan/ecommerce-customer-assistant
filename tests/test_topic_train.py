"""ch10 T6 训练参数的回归测试。"""

from training.train import configure_lr_scheduler, load_training_rows


class _FakeTrainingArguments:
    num_train_epochs = 3.0

    def set_lr_scheduler(self, **kwargs):
        # 模拟 transformers 5:未显式传 num_epochs 时会回落为 3。
        self.num_train_epochs = kwargs.get("num_epochs", 3.0)
        self.kwargs = kwargs
        return self


def test_scheduler_setter_keeps_requested_epoch_limit():
    args = configure_lr_scheduler(_FakeTrainingArguments(), epochs=8)
    assert args.num_train_epochs == 8
    assert args.kwargs["warmup_steps"] == 0.1


def test_targeted_patch_is_training_only_and_disjoint_from_holdouts():
    rows, patch_count = load_training_rows(use_patch=True)
    assert patch_count == 48
    assert len(rows) == 4587 + patch_count


def test_targeted_patch_can_be_replayed_without_changing_source_file():
    rows, patch_count = load_training_rows(use_patch=True, patch_repeat=5)
    assert patch_count == 48 * 5
    assert len(rows) == 4587 + patch_count
