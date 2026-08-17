# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LanguageWer:
    wer: float
    errors: int
    reference_words: int
    count: int


@dataclass(frozen=True)
class ValidationRun:
    name: str
    path: Path
    label: str
    overall: LanguageWer
    by_language: dict[str, LanguageWer]


LANGUAGE_LABELS = {
    "de_de": "German",
    "en_us": "English (US)",
    "es_419": "Spanish (LatAm)",
    "fr_fr": "French",
    "it_it": "Italian",
}


def main() -> None:
    args = parse_args()
    runs = discover_runs(args.runs_dir, args.include, args.exclude, args.min_total_count)
    if not runs:
        raise SystemExit(
            f"No validation summaries found in {args.runs_dir} after applying filters. "
            "Try lowering --min-total-count or passing --include."
        )

    baseline = select_baseline(runs, args.baseline)
    ordered_runs = [baseline, *sorted((run for run in runs if run != baseline), key=lambda run: run.overall.wer)]
    ordered_runs = with_unique_labels(ordered_runs)
    baseline = ordered_runs[0]
    languages = order_languages(ordered_runs, baseline)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_runs(ordered_runs, baseline, languages, args.output)

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(ordered_runs, baseline, languages, args.csv)

    print(f"Wrote {args.output}")
    if args.csv is not None:
        print(f"Wrote {args.csv}")
    print(f"Baseline: {baseline.name} ({baseline.overall.wer:.2%} WER, n={baseline.overall.count})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-language WER and relative WER differences from rnnt.validation summaries."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("artifacts/validation/runs"),
        help="Directory containing validation run subdirectories with summary.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/validation/wer_by_language.png"),
        help="Output image path. The extension controls the Matplotlib output format.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("artifacts/validation/wer_by_language.csv"),
        help="Optional CSV output path with the plotted values. Use --csv '' to disable.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Baseline run name or path. Defaults to the run with the lowest overall WER after filtering.",
    )
    parser.add_argument(
        "--include",
        default=None,
        help="Regex matched against run names. Only matching runs are included.",
    )
    parser.add_argument(
        "--exclude",
        default=None,
        help="Regex matched against run names. Matching runs are excluded.",
    )
    parser.add_argument(
        "--min-total-count",
        type=int,
        default=1000,
        help="Minimum total utterance count required for a run. Defaults to 1000 to hide smoke runs.",
    )
    args = parser.parse_args()
    if args.csv == Path(""):
        args.csv = None
    return args


def discover_runs(
    runs_dir: Path,
    include: str | None,
    exclude: str | None,
    min_total_count: int,
) -> list[ValidationRun]:
    include_re = re.compile(include) if include else None
    exclude_re = re.compile(exclude) if exclude else None
    runs = []
    for summary_path in sorted(runs_dir.glob("*/summary.json")):
        run_name = summary_path.parent.name
        if include_re is not None and include_re.search(run_name) is None:
            continue
        if exclude_re is not None and exclude_re.search(run_name) is not None:
            continue

        run = load_run(summary_path)
        if run.overall.count < min_total_count:
            continue
        runs.append(run)
    return runs


def load_run(summary_path: Path) -> ValidationRun:
    with summary_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    summary = payload["summary"]
    overall = LanguageWer(
        wer=float(summary["wer"]),
        errors=int(summary["errors"]),
        reference_words=int(summary["reference_words"]),
        count=int(summary["count"]),
    )
    by_language = {
        language: LanguageWer(
            wer=float(values["wer"]),
            errors=int(values["errors"]),
            reference_words=int(values["reference_words"]),
            count=int(values["count"]),
        )
        for language, values in summary.get("by_language", {}).items()
    }
    return ValidationRun(
        summary_path.parent.name,
        summary_path,
        concise_label(summary_path.parent.name, payload.get("config", {})),
        overall,
        by_language,
    )


def select_baseline(runs: list[ValidationRun], baseline: str | None) -> ValidationRun:
    if baseline is None:
        return min(runs, key=lambda run: run.overall.wer)

    baseline_path = Path(baseline)
    for run in runs:
        if run.name == baseline or run.path.parent == baseline_path or run.path == baseline_path:
            return run
    choices = ", ".join(run.name for run in runs)
    raise SystemExit(f"Baseline {baseline!r} was not found after filtering. Available runs: {choices}")


def order_languages(runs: list[ValidationRun], baseline: ValidationRun) -> list[str]:
    languages = sorted({language for run in runs for language in run.by_language})
    return sorted(
        languages,
        key=lambda language: baseline.by_language.get(language, LanguageWer(0.0, 0, 0, 0)).wer,
        reverse=True,
    )


def with_unique_labels(runs: list[ValidationRun]) -> list[ValidationRun]:
    label_counts: dict[str, int] = {}
    unique_runs = []
    for run in runs:
        label_counts[run.label] = label_counts.get(run.label, 0) + 1
        label = run.label if label_counts[run.label] == 1 else f"{run.label} #{label_counts[run.label]}"
        unique_runs.append(ValidationRun(run.name, run.path, label, run.overall, run.by_language))
    return unique_runs


def concise_label(name: str, config: dict) -> str:
    backend = config.get("backend")
    dtype = config.get("dtype")
    attention = config.get("attention")
    quantize = config.get("quantize")

    if backend == "cpp":
        provider = config.get("cpp_provider") or "cpp"
        return f"C++ {provider.upper()}"
    if quantize and quantize != "none":
        bits = config.get("auto_quant_effective_bits")
        bits_suffix = f" {bits:g}b" if isinstance(bits, (int, float)) else ""
        return f"{quantize.replace('-', ' ').title().replace('Int4', 'INT4')}{bits_suffix}"
    if attention == "rel_pos_local_attn":
        context = str(config.get("att_context_size") or "").replace(",", "/")
        context_suffix = f" {context}" if context else ""
        dtype_suffix = f" {dtype}" if dtype else ""
        return f"Local attn{context_suffix}{dtype_suffix}"
    if dtype:
        return f"Full attn {dtype}"
    return name.replace("_", " ")


def plot_runs(
    runs: list[ValidationRun],
    baseline: ValidationRun,
    languages: list[str],
    output: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "Plotting requires matplotlib. Install it with `python -m pip install -e .[validation]`."
        ) from error

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "font.size": 10,
            "legend.frameon": False,
        }
    )

    width = max(11.5, 1.1 * len(languages) + 4.0)
    height = 7.0 if len(runs) > 1 else 4.8
    figure, axes = plt.subplots(
        2 if len(runs) > 1 else 1,
        1,
        figsize=(width, height),
        gridspec_kw={"height_ratios": [1.2, 1.0]} if len(runs) > 1 else None,
    )
    figure.subplots_adjust(top=0.82, hspace=0.34, left=0.13, right=0.98, bottom=0.08)
    if len(runs) == 1:
        wer_axis = axes
        delta_axis = None
    else:
        wer_axis, delta_axis = axes

    y_positions = list(range(len(languages)))
    bar_height = min(0.72 / len(runs), 0.18)
    offsets = centered_offsets(len(runs), bar_height)
    colors = palette(plt, runs, baseline)

    for index, run in enumerate(runs):
        values = [percent(run.by_language.get(language)) for language in languages]
        wer_axis.barh(
            [position + offsets[index] for position in y_positions],
            values,
            height=bar_height,
            label=legend_label(run, baseline),
            color=colors[run.name],
            alpha=0.92,
        )

    wer_axis.set_title("WER by language")
    wer_axis.set_xlabel("WER (%)")
    wer_axis.set_yticks(y_positions, [language_label(language) for language in languages])
    wer_axis.invert_yaxis()
    wer_axis.grid(axis="x", linestyle="-", alpha=0.18)
    wer_axis.set_axisbelow(True)
    annotate_absolute_values(wer_axis, runs, languages, y_positions, offsets, colors)

    if delta_axis is not None:
        comparison_runs = [run for run in runs if run != baseline]
        delta_height = min(0.72 / max(len(comparison_runs), 1), 0.22)
        delta_offsets = centered_offsets(len(comparison_runs), delta_height)
        for index, run in enumerate(comparison_runs):
            deltas = [
                relative_delta_percent(run.by_language.get(language), baseline.by_language.get(language))
                for language in languages
            ]
            delta_axis.barh(
                [position + delta_offsets[index] for position in y_positions],
                deltas,
                height=delta_height,
                label=legend_label(run, baseline),
                color=colors[run.name],
                alpha=0.92,
            )

        delta_axis.axvline(0.0, color="#303030", linewidth=1.0)
        delta_axis.set_title(f"Relative WER change vs {baseline.label}")
        delta_axis.set_xlabel("Change in WER (%)")
        delta_axis.set_yticks(y_positions, [language_label(language) for language in languages])
        delta_axis.invert_yaxis()
        delta_axis.grid(axis="x", linestyle="-", alpha=0.18)
        delta_axis.set_axisbelow(True)
        annotate_delta_values(delta_axis, comparison_runs, baseline, languages, y_positions, delta_offsets)

    handles, labels = wer_axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.9),
        ncols=min(len(runs), 4),
    )
    figure.suptitle("Validation WER Comparison", fontsize=16, fontweight="bold", y=0.98)
    figure.text(
        0.5,
        0.94,
        f"Baseline: {baseline.label} ({baseline.overall.wer:.2%} overall WER, n={baseline.overall.count}). Lower is better.",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def centered_offsets(count: int, width: float) -> list[float]:
    center = (count - 1) / 2
    return [(index - center) * width for index in range(count)]


def language_label(language: str) -> str:
    return LANGUAGE_LABELS.get(language, language)


def palette(plt, runs: list[ValidationRun], baseline: ValidationRun) -> dict[str, str]:
    color_cycle = list(plt.get_cmap("Set2").colors) + list(plt.get_cmap("tab10").colors)
    colors = {baseline.name: "#2f3437"}
    color_index = 0
    for run in runs:
        if run == baseline:
            continue
        colors[run.name] = color_cycle[color_index % len(color_cycle)]
        color_index += 1
    return colors


def legend_label(run: ValidationRun, baseline: ValidationRun) -> str:
    prefix = "Baseline: " if run == baseline else ""
    return f"{prefix}{run.label} ({run.overall.wer:.2%})"


def annotate_absolute_values(
    axis,
    runs: list[ValidationRun],
    languages: list[str],
    y_positions: list[int],
    offsets: list[float],
    colors: dict[str, str],
) -> None:
    max_value = max(
        percent(run.by_language.get(language))
        for run in runs
        for language in languages
        if run.by_language.get(language) is not None
    )
    axis.set_xlim(0, max_value * 1.16)
    for run_index, run in enumerate(runs):
        for y_position, language in zip(y_positions, languages, strict=True):
            value = percent(run.by_language.get(language))
            if math.isnan(value):
                continue
            axis.text(
                value + max_value * 0.012,
                y_position + offsets[run_index],
                f"{value:.1f}",
                va="center",
                ha="left",
                fontsize=8,
                color=colors[run.name],
            )


def annotate_delta_values(
    axis,
    runs: list[ValidationRun],
    baseline: ValidationRun,
    languages: list[str],
    y_positions: list[int],
    offsets: list[float],
) -> None:
    visible_values = []
    for run in runs:
        for language in languages:
            value = relative_delta_percent(run.by_language.get(language), baseline.by_language.get(language))
            if not math.isnan(value):
                visible_values.append(value)

    if not visible_values:
        return

    min_value = min(visible_values)
    max_value = max(visible_values)
    max_abs = max(abs(min_value), abs(max_value), 1.0)
    axis.set_xlim(min(min_value * 1.2, -0.12 * max_abs), max(max_value * 1.2, 0.12 * max_abs))
    axis.axvspan(axis.get_xlim()[0], 0, color="#2ca25f", alpha=0.06, zorder=0)
    axis.axvspan(0, axis.get_xlim()[1], color="#de2d26", alpha=0.05, zorder=0)

    label_offset = max_abs * 0.025
    for run_index, run in enumerate(runs):
        for y_position, language in zip(y_positions, languages, strict=True):
            value = relative_delta_percent(run.by_language.get(language), baseline.by_language.get(language))
            if math.isnan(value):
                continue
            axis.text(
                value + (label_offset if value >= 0 else -label_offset),
                y_position + offsets[run_index],
                f"{value:+.1f}%",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=8,
                color="#404040",
            )


def percent(value: LanguageWer | None) -> float:
    if value is None:
        return math.nan
    return 100.0 * value.wer


def relative_delta_percent(value: LanguageWer | None, baseline: LanguageWer | None) -> float:
    if value is None or baseline is None or baseline.wer == 0.0:
        return math.nan
    return 100.0 * (value.wer - baseline.wer) / baseline.wer


def write_csv(
    runs: list[ValidationRun],
    baseline: ValidationRun,
    languages: list[str],
    csv_path: Path,
) -> None:
    fields = [
        "run",
        "language",
        "wer",
        "wer_percent",
        "errors",
        "reference_words",
        "count",
        "baseline_run",
        "baseline_wer",
        "absolute_delta",
        "relative_delta_percent",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            write_csv_row(writer, run, "__overall__", run.overall, baseline.name, baseline.overall)
            for language in languages:
                value = run.by_language.get(language)
                baseline_value = baseline.by_language.get(language)
                if value is None:
                    continue
                write_csv_row(writer, run, language, value, baseline.name, baseline_value)


def write_csv_row(
    writer: csv.DictWriter,
    run: ValidationRun,
    language: str,
    value: LanguageWer,
    baseline_name: str,
    baseline: LanguageWer | None,
) -> None:
    writer.writerow(
        {
            "run": run.name,
            "language": language,
            "wer": value.wer,
            "wer_percent": percent(value),
            "errors": value.errors,
            "reference_words": value.reference_words,
            "count": value.count,
            "baseline_run": baseline_name,
            "baseline_wer": baseline.wer if baseline is not None else "",
            "absolute_delta": value.wer - baseline.wer if baseline is not None else "",
            "relative_delta_percent": relative_delta_percent(value, baseline),
        }
    )


if __name__ == "__main__":
    main()
