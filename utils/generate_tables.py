import argparse
import json
import math
import os
import sys

def load(path):
    if not os.path.exists(path):
        print(f"[ERRO] Arquivo não encontrado: {path}")
        print("Execute main_av3.py primeiro para gerar os resultados.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)

def _ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception:
        print("[ERRO] matplotlib não encontrado.")
        print("Instale com: pip install matplotlib")
        sys.exit(1)
    return plt, PdfPages


def _fmt(val, digits=4):
    if isinstance(val, float):
        return f"{val:.{digits}f}"
    return str(val)


def _draw_table(ax, col_labels, rows, font_size=9, col_widths=None, row_scale=1.2):
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, row_scale)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#0d1b2a")
        else:
            cell.set_edgecolor("#e2e8f0")
    return table


def _highlight_cells(table, highlights):
    for row, col, face_color, text_color in highlights:
        cell = table[(row + 1, col)]
        cell.set_facecolor(face_color)
        if text_color:
            cell.get_text().set_color(text_color)
            cell.get_text().set_weight("bold")


def _save_pdf_page(pdf, fig):
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)


def _plot_clf_summary(pdf, plt, data):
    labels = [f"LR {r['lr']}\n{r['epochs']} ep" for r in data]
    acc = [r["accuracy"] for r in data]
    f1 = [r["f1"] for r in data]

    fig, (ax_chart, ax_table) = plt.subplots(
        2, 1, figsize=(11.69, 8.27), gridspec_kw={"height_ratios": [2.2, 1]}
    )

    x = list(range(len(data)))
    width = 0.35
    ax_chart.bar([i - width / 2 for i in x], acc, width, label="Acurácia")
    ax_chart.bar([i + width / 2 for i in x], f1, width, label="F1-Score")
    ax_chart.set_xticks(x)
    ax_chart.set_xticklabels(labels)
    ax_chart.set_ylim(0, 1)
    ax_chart.set_ylabel("Score")
    ax_chart.grid(axis="y", alpha=0.3)
    ax_chart.legend()

    col_labels = [
        "LR",
        "Épocas",
        "Acurácia",
        "Precisão",
        "Recall",
        "Especificidade",
        "F1",
        "Tempo (s)",
    ]
    rows = [
        [
            _fmt(r["lr"], 3),
            r["epochs"],
            _fmt(r["accuracy"]),
            _fmt(r["precision"]),
            _fmt(r["recall"]),
            _fmt(r["specificity"]),
            _fmt(r["f1"]),
            _fmt(r["time"]),
        ]
        for r in data
    ]
    table = _draw_table(ax_table, col_labels, rows, font_size=9, row_scale=1.2)

    best_acc = max(acc)
    worst_acc = min(acc)
    best_f1 = max(f1)
    highlights = []
    for i, r in enumerate(data):
        if abs(r["accuracy"] - best_acc) < 1e-9:
            highlights.append((i, 2, "#d1fae5", "#065f46"))
        if abs(r["accuracy"] - worst_acc) < 1e-9:
            highlights.append((i, 2, "#fee2e2", "#991b1b"))
        if abs(r["f1"] - best_f1) < 1e-9:
            highlights.append((i, 6, "#d1fae5", "#065f46"))
    _highlight_cells(table, highlights)

    fig.suptitle(
        "Perceptron OvA — Classificação (Students Dropout)",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(0.5, 0.95, "K-Fold 5x", ha="center", fontsize=10, color="#64748b")
    _save_pdf_page(pdf, fig)
    plt.close(fig)


def _plot_clf_confusion_matrices(pdf, plt, data):
    cms = [
        (r["lr"], r["epochs"], r["confusion_matrix"])
        for r in data
        if r.get("confusion_matrix")
    ]
    if not cms:
        return

    cols = 2
    rows = int(math.ceil(len(cms) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11.69, 8.27))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, (lr, epochs, cm) in zip(axes_list, cms):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"LR {lr} / {epochs} ep")
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(["Pred 0", "Pred 1", "Pred 2"])
        ax.set_yticklabels(["Real 0", "Real 1", "Real 2"])
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i][j], ha="center", va="center", color="#0f172a")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes_list[len(cms):]:
        ax.axis("off")

    fig.suptitle(
        "Matrizes de Confusão — Perceptron (último fold)",
        fontsize=14,
        fontweight="bold",
    )
    _save_pdf_page(pdf, fig)
    plt.close(fig)


def _plot_reg_summary(pdf, plt, data):
    sorted_data = sorted(data, key=lambda r: r["r2"], reverse=True)
    top_n = min(10, len(sorted_data))
    top = sorted_data[:top_n]

    fig, (ax_chart, ax_table) = plt.subplots(
        2, 1, figsize=(11.69, 8.27), gridspec_kw={"height_ratios": [2.2, 1]}
    )

    labels = [f"#{i}" for i in range(1, top_n + 1)]
    r2_vals = [r["r2"] for r in top]
    ax_chart.bar(labels, r2_vals, color="#1b998b")
    ax_chart.axhline(0, color="#0f172a", linewidth=0.8)
    ax_chart.set_ylabel("R²")
    ax_chart.set_title("Top R² — MLP Regressor")
    ax_chart.grid(axis="y", alpha=0.3)

    col_labels = [
        "#",
        "Rodada",
        "Topologia",
        "Ativação",
        "LR",
        "Épocas",
        "R²",
        "RMSE",
        "MAE",
    ]
    rows = [
        [
            i,
            r["rodada"],
            r["topology"],
            r["activation"],
            _fmt(r["lr"], 3),
            r["epochs"],
            _fmt(r["r2"]),
            _fmt(r["rmse"]),
            _fmt(r["mae"]),
        ]
        for i, r in enumerate(top, 1)
    ]
    _draw_table(ax_table, col_labels, rows, font_size=9, row_scale=1.2)

    fig.suptitle(
        "MLP Regressor — Regressão (Dataset Anime)",
        fontsize=14,
        fontweight="bold",
    )
    _save_pdf_page(pdf, fig)
    plt.close(fig)


def _plot_reg_best_by_topology(pdf, plt, data):
    topologies = sorted(set(r["topology"] for r in data))
    rows = []
    for top in topologies:
        subset = [r for r in data if r["topology"] == top]
        best = max(subset, key=lambda r: r["r2"])
        rows.append(
            [
                top,
                best["rodada"],
                best["activation"],
                _fmt(best["lr"], 3),
                best["epochs"],
                _fmt(best["r2"]),
                _fmt(best["rmse"]),
                _fmt(best["train_time"]),
            ]
        )

    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    col_labels = [
        "Topologia",
        "Rodada",
        "Ativação",
        "LR",
        "Épocas",
        "R²",
        "RMSE",
        "T. Treino (s)",
    ]
    _draw_table(ax, col_labels, rows, font_size=10, row_scale=1.3)
    fig.suptitle("Melhor Resultado por Topologia", fontsize=14, fontweight="bold")
    _save_pdf_page(pdf, fig)
    plt.close(fig)


def _plot_reg_full_tables(pdf, plt, data, rows_per_page=24):
    sorted_data = sorted(data, key=lambda r: r["r2"], reverse=True)
    col_labels = [
        "#",
        "Rodada",
        "Topologia",
        "Ativação",
        "LR",
        "Épocas",
        "R²",
        "R² Adj.",
        "RMSE",
        "MAE",
        "MSE",
        "T. Treino (s)",
        "T. Teste (s)",
    ]
    for start in range(0, len(sorted_data), rows_per_page):
        chunk = sorted_data[start : start + rows_per_page]
        rows = []
        for idx, r in enumerate(chunk, start + 1):
            rows.append(
                [
                    idx,
                    r["rodada"],
                    r["topology"],
                    r["activation"],
                    _fmt(r["lr"], 3),
                    r["epochs"],
                    _fmt(r["r2"]),
                    _fmt(r["r2_adj"]),
                    _fmt(r["rmse"]),
                    _fmt(r["mae"]),
                    _fmt(r["mse"]),
                    _fmt(r["train_time"]),
                    _fmt(r["test_time"], 6),
                ]
            )

        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        _draw_table(ax, col_labels, rows, font_size=7, row_scale=1.2)
        fig.suptitle(
            f"Tabela Completa — MLP Regressor (linhas {start + 1}-{start + len(chunk)})",
            fontsize=13,
            fontweight="bold",
        )
        _save_pdf_page(pdf, fig)
        plt.close(fig)


def build_pdf_report(clf_data, reg_data, out_path):
    plt, PdfPages = _ensure_matplotlib()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with PdfPages(out_path) as pdf:
        _plot_clf_summary(pdf, plt, clf_data)
        _plot_clf_confusion_matrices(pdf, plt, clf_data)
        _plot_reg_summary(pdf, plt, reg_data)
        _plot_reg_best_by_topology(pdf, plt, reg_data)
        _plot_reg_full_tables(pdf, plt, reg_data)

    print(f"✓ Relatório gerado em: {out_path}")
    print("  Abra o PDF para visualizar.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera um relatório em PDF com gráficos e tabelas.")
    parser.add_argument(
        "--out",
        default=None,
        help="Caminho de saída (opcional).",
    )
    args = parser.parse_args()

    clf_data = load("results/clf_results.json")
    reg_data = load("results/reg_results.json")

    os.makedirs("results", exist_ok=True)
    out = args.out or "results/relatorio_resultados.pdf"
    build_pdf_report(clf_data, reg_data, out)

