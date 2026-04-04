"""预处理脚本进度显示工具。"""

from __future__ import annotations

from tqdm import tqdm


def log_stage(message: str) -> None:
    print(f"\n[阶段] {message}", flush=True)


class ProgressBar:
    """基于 tqdm 的轻量封装。"""

    def __init__(self, label: str, total: int | None = None, unit: str = "项") -> None:
        self.unit = unit
        self.current = 0
        self.progress = tqdm(
            total=total,
            desc=label,
            unit=unit,
            dynamic_ncols=True,
            leave=True,
        )

    def update(self, step: int = 1, detail: str | None = None) -> None:
        self.current += step
        if detail:
            self.progress.set_postfix_str(f"当前: {detail}")
        if step:
            self.progress.update(step)
        else:
            self.progress.refresh()

    def finish(self, detail: str | None = None) -> None:
        if detail:
            self.update(step=0, detail=detail)
        self.progress.close()
