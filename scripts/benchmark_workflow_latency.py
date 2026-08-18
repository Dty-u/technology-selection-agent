"""Read-only latency benchmark for the compatibility workflow graph."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from statistics import median
from time import perf_counter
from uuid import uuid4

from src.config.bootstrap import warm_technology_selection_resources
from src.workflows.query_router_graph import query_router_graph
from src.workflows.selection_cache import clear_selection_cache


SAMPLES = {
    "direct": (
        ("greeting", "你好"),
        ("capabilities", "你能做什么？"),
        ("out_of_scope", "帮我写一首关于海边的诗。"),
    ),
    "local": (
        ("gaa_finfet", "解释 GAA 与 FinFET 的结构差异和影响。"),
        ("euv_defects", "EUV 光刻常见缺陷的成因是什么？"),
        ("tsv_hybrid_bonding", "比较 TSV 与混合键合的关键工艺特点。"),
    ),
    "web": (
        ("2nm_production", "截至目前 2nm 工艺的量产进展是什么？"),
        ("sic_suppliers", "目前有哪些 8 英寸 SiC 晶圆供应商？"),
        ("hbm4_roadmap", "截至目前 HBM4 的公开路线图如何？"),
    ),
    "selection": (
        (
            "ald_cvd",
            "在高深宽比沟槽沉积中选择 ALD 或 CVD，目标是提高覆盖率并控制成本。",
        ),
        (
            "finfet_gaa",
            "为低功耗手机 SoC 选择 FinFET 或 GAA，目标是降低功耗并保证性能。",
        ),
        (
            "sic_gan",
            "为 800V 电驱逆变器选择 SiC 或 GaN，目标是提高效率和量产可靠性。",
        ),
    ),
}


def summarize(records):
    summary = {}
    for path in SAMPLES:
        path_records = [record for record in records if record.get("path") == path]
        timings = [float(record["elapsed_seconds"]) for record in path_records]
        if timings:
            terminal_count = sum(
                record.get("status") in {"completed", "needs_human_review"}
                for record in path_records
            )
            summary[path] = {
                "sample_count": len(timings),
                "terminal_sample_count": terminal_count,
                "comparison_eligible": (
                    len(path_records) == len(SAMPLES[path])
                    and terminal_count == len(path_records)
                ),
                "median_seconds": round(median(timings), 3),
                "max_seconds": round(max(timings), 3),
            }
    return summary


def run_benchmark(paths):
    warmup = warm_technology_selection_resources()
    records = []
    for path in paths:
        for sample_id, query in SAMPLES[path]:
            if path == "selection":
                clear_selection_cache()
            request_id = str(uuid4())
            config = {
                "configurable": {
                    "user_id": "latency-benchmark",
                    "thread_id": f"latency-{path}-{sample_id}-{request_id}",
                    "request_id": request_id,
                    "project_id": "latency-benchmark",
                }
            }
            started_at = perf_counter()
            try:
                result = query_router_graph.invoke({"query": query}, config=config)
                record = {
                    "path": path,
                    "sample_id": sample_id,
                    "elapsed_seconds": round(perf_counter() - started_at, 3),
                    "status": result.get("status", "unknown"),
                    "query_type": result.get("query_type", "unknown"),
                }
            except Exception as exc:
                record = {
                    "path": path,
                    "sample_id": sample_id,
                    "elapsed_seconds": round(perf_counter() - started_at, 3),
                    "status": "error",
                    "error_type": type(exc).__name__,
                }
            records.append(record)
    return {"warmup": warmup, "records": records, "summary": summarize(records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths",
        nargs="+",
        choices=tuple(SAMPLES),
        default=list(SAMPLES),
    )
    parser.add_argument("--label", default="after")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = {
        "label": args.label,
        "created_at": datetime.now().astimezone().isoformat(),
        **run_benchmark(args.paths),
    }
    output = args.output or Path("runtime/reports") / (
        f"latency-{args.label}-{datetime.now():%Y%m%d-%H%M%S}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(output.resolve())


if __name__ == "__main__":
    main()
