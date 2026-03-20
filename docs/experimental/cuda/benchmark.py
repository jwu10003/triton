"""CUDA Kernel 性能自动测量工具。

LLM Agent 生成 kernel 后调用此模块测量耗时，获取结构化性能数据用于优化决策。
使用 CUDA Event 计时，精确测量 GPU 时间，不含 CPU 开销。

用法:
    from benchmark import benchmark, benchmark_many

    result = benchmark(lambda: kernel(args...), warmup=25, rep=100)
    results = benchmark_many({"v1": fn1, "v2": fn2}, quantiles=[0.5, 0.9])
"""

import json
import statistics
from typing import Any, Callable, Dict, List, Optional

import torch


def benchmark(
    fn: Callable,
    warmup: int = 25,
    rep: int = 100,
    quantiles: Optional[List[float]] = None,
    return_mode: str = "dict",
    label: str = "",
    sync_before: bool = True,
) -> Dict[str, Any]:
    """测量单个 CUDA kernel 的执行时间。

    Args:
        fn: 待测量的 callable (调用方已包装好，如 lambda: kernel(args...))。
        warmup: 预热次数，让 GPU 进入稳态。
        rep: 重复测量次数。
        quantiles: 需要计算的分位数列表，如 [0.1, 0.5, 0.9]。
        return_mode: 返回模式。
            "dict" - 返回结果字典。
            "json" - 返回结果字典，同时打印 JSON。
            "print" - 返回结果字典，同时打印人类可读表格。
        label: 标签名，用于输出标识。
        sync_before: 测量前是否 synchronize，清空 GPU 队列。

    Returns:
        包含 mean/median/min/max/std/cv/quantiles/all_times 等字段的字典。
    """
    assert rep > 0, f"rep must be > 0, got {rep}"
    assert warmup >= 0, f"warmup must be >= 0, got {warmup}"

    # 1. 清空队列
    if sync_before:
        torch.cuda.synchronize()

    # 2. 预热
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # 3. 测量：每次迭代单独 record + synchronize，避免 event queue 堆积
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    times: List[float] = []

    for _ in range(rep):
        start_event.record()
        fn()
        end_event.record()
        end_event.synchronize()
        times.append(start_event.elapsed_time(end_event))  # 毫秒

    # 4. 统计
    mean = statistics.mean(times)
    std = statistics.stdev(times) if rep > 1 else 0.0

    result: Dict[str, Any] = {
        "label": label,
        "warmup": warmup,
        "rep": rep,
        "unit": "ms",
        "mean": mean,
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "std": std,
        "cv": (std / mean * 100) if mean > 0 else 0.0,
    }

    # 分位数
    if quantiles:
        sorted_times = sorted(times)
        n = len(sorted_times)
        q_dict = {}
        for q in quantiles:
            assert 0.0 <= q <= 1.0, f"quantile must be in [0, 1], got {q}"
            idx = min(int(q * n), n - 1)
            q_dict[f"p{int(q * 100)}"] = sorted_times[idx]
        result["quantiles"] = q_dict

    result["all_times"] = times

    # 输出
    if return_mode == "json":
        # all_times 可能很长，JSON 输出时保留
        print(json.dumps(result, indent=2))
    elif return_mode == "print":
        print_result(result)

    return result


def benchmark_many(
    fns: Dict[str, Callable],
    warmup: int = 25,
    rep: int = 100,
    quantiles: Optional[List[float]] = None,
    return_mode: str = "dict",
    sync_before: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """批量测量多个 kernel，返回对比结果。

    Args:
        fns: {名称: callable} 字典。
        warmup: 预热次数。
        rep: 重复测量次数。
        quantiles: 分位数列表。
        return_mode: "dict" | "json" | "print"。
        sync_before: 测量前是否 synchronize。

    Returns:
        {名称: 结果字典} 的字典。
    """
    results = {}
    for name, fn in fns.items():
        results[name] = benchmark(
            fn,
            warmup=warmup,
            rep=rep,
            quantiles=quantiles,
            return_mode="dict",  # 内部不打印，最后统一输出
            label=name,
            sync_before=sync_before,
        )

    if return_mode == "json":
        print(json.dumps(results, indent=2))
    elif return_mode == "print":
        print_comparison(results)

    return results


def print_result(result: Dict[str, Any]) -> None:
    """格式化打印单个 benchmark 结果。"""
    label = result.get("label", "")
    header = f"[benchmark] {label}" if label else "[benchmark]"
    print(header)
    print(f"  warmup: {result['warmup']}, rep: {result['rep']}")
    print(f"  mean    = {result['mean']:.3f} ms")
    print(f"  median  = {result['median']:.3f} ms")
    print(f"  min     = {result['min']:.3f} ms")
    print(f"  max     = {result['max']:.3f} ms")
    print(f"  std     = {result['std']:.3f} ms (cv={result['cv']:.2f}%)")

    if "quantiles" in result:
        for qname, qval in result["quantiles"].items():
            print(f"  {qname:<7} = {qval:.3f} ms")


def print_comparison(results: Dict[str, Dict[str, Any]]) -> None:
    """打印多个 kernel 的对比表格。

    以第一个 kernel 的 mean 为基准计算 speedup。
    """
    if not results:
        return

    names = list(results.keys())
    baseline_mean = results[names[0]]["mean"]

    # 列宽
    name_w = max(len(n) for n in names)
    name_w = max(name_w, 6)  # 最小 "kernel"
    val_w = 8

    def _fmt(v: float) -> str:
        return f"{v:.3f}"

    # 表头
    hdr = (
        f"│ {'kernel':<{name_w}} "
        f"│ {'mean(ms)':>{val_w}} "
        f"│ {'min(ms)':>{val_w}} "
        f"│ {'max(ms)':>{val_w}} "
        f"│ {'std(ms)':>{val_w}} "
        f"│ {'speedup':>{val_w}} │"
    )
    sep_mid = (
        f"├{'─' * (name_w + 2)}"
        f"┼{'─' * (val_w + 2)}"
        f"┼{'─' * (val_w + 2)}"
        f"┼{'─' * (val_w + 2)}"
        f"┼{'─' * (val_w + 2)}"
        f"┼{'─' * (val_w + 2)}┤"
    )
    sep_top = sep_mid.replace("├", "┌").replace("┼", "┬").replace("┤", "┐")
    sep_bot = sep_mid.replace("├", "└").replace("┼", "┴").replace("┤", "┘")

    print(sep_top)
    print(hdr)
    print(sep_mid)

    for name in names:
        r = results[name]
        speedup = baseline_mean / r["mean"] if r["mean"] > 0 else float("inf")
        row = (
            f"│ {name:<{name_w}} "
            f"│ {_fmt(r['mean']):>{val_w}} "
            f"│ {_fmt(r['min']):>{val_w}} "
            f"│ {_fmt(r['max']):>{val_w}} "
            f"│ {_fmt(r['std']):>{val_w}} "
            f"│ {f'{speedup:.2f}x':>{val_w}} │"
        )
        print(row)

    print(sep_bot)


if __name__ == "__main__":
    # 简单自测
    def _dummy():
        a = torch.randn(1024, 1024, device="cuda")
        b = torch.randn(1024, 1024, device="cuda")
        torch.mm(a, b)

    print("=== Single benchmark ===")
    result = benchmark(
        _dummy, warmup=5, rep=20, quantiles=[0.5, 0.9], return_mode="print"
    )
    assert "mean" in result
    assert "median" in result
    assert result["rep"] == 20
    print()

    print("=== Comparison ===")
    benchmark_many(
        {"matmul_1k": _dummy, "matmul_1k_again": _dummy},
        warmup=5,
        rep=20,
        quantiles=[0.5, 0.9],
        return_mode="print",
    )
