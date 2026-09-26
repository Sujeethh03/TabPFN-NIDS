"""
PCAP Analysis Report Generator

Creates JSON and HTML reports for individual PCAP analysis runs.
"""

import json
from pathlib import Path
from datetime import datetime


def _safe_value(value):
    """Convert values into JSON-safe Python values."""

    if hasattr(value, "item"):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    return value


def save_json_report(
    report_data: dict,
    output_path: Path,
) -> Path:
    """Save the complete analysis report as JSON."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report_data,
            f,
            indent=2,
            default=_safe_value,
        )

    return output_path


def _html_escape(value):
    """Escape text for safe HTML output."""

    text = str(value)

    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _table_rows(data: dict) -> str:
    """Convert a dictionary into HTML table rows."""

    rows = []

    for key, value in data.items():

        if isinstance(value, float):
            value = f"{value:.4f}"

        rows.append(
            f"""
            <tr>
                <td>{_html_escape(key)}</td>
                <td>{_html_escape(value)}</td>
            </tr>
            """
        )

    return "\n".join(rows)


def save_html_report(
    report_data: dict,
    output_path: Path,
) -> Path:
    """Save the complete analysis report as HTML."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pcap = report_data.get(
        "pcap",
        {},
    )

    extraction = report_data.get(
        "extraction",
        {},
    )

    features = report_data.get(
        "features",
        {},
    )

    model = report_data.get(
        "model",
        {},
    )

    predictions = report_data.get(
        "predictions",
        {},
    )

    ground_truth = report_data.get(
        "ground_truth"
    )

    evaluation = report_data.get(
        "evaluation"
    )

    inference = report_data.get(
        "inference"
    )

    generated_at = report_data.get(
        "generated_at",
        datetime.now().isoformat(),
    )

    html = f"""<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>PCAP Analysis Report</title>

<style>

body {{
    font-family:
        Arial,
        Helvetica,
        sans-serif;

    margin: 0;
    padding: 0;

    background: #f5f7fa;
    color: #1f2937;
}}

.container {{
    max-width: 1100px;

    margin: 40px auto;

    background: white;

    padding: 40px;

    border-radius: 12px;
}}

h1 {{
    margin-bottom: 5px;
}}

h2 {{
    margin-top: 35px;

    border-bottom:
        2px solid #e5e7eb;

    padding-bottom: 8px;
}}

.subtitle {{
    color: #6b7280;

    margin-bottom: 30px;
}}

.summary {{
    display: grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(180px, 1fr)
        );

    gap: 15px;

    margin: 25px 0;
}}

.card {{
    background: #f9fafb;

    border:
        1px solid #e5e7eb;

    border-radius: 10px;

    padding: 18px;
}}

.card-title {{
    font-size: 13px;

    color: #6b7280;
}}

.card-value {{
    font-size: 25px;

    font-weight: bold;

    margin-top: 5px;
}}

table {{
    width: 100%;

    border-collapse: collapse;

    margin-top: 15px;
}}

th,
td {{
    text-align: left;

    padding: 10px;

    border-bottom:
        1px solid #e5e7eb;
}}

th {{
    background: #f3f4f6;
}}

.footer {{
    margin-top: 40px;

    padding-top: 20px;

    border-top:
        1px solid #e5e7eb;

    color: #6b7280;

    font-size: 13px;
}}

</style>

</head>

<body>

<div class="container">

<h1>
    PCAP Analysis Report
</h1>

<div class="subtitle">
    Generated:
    {_html_escape(generated_at)}
</div>


<div class="summary">


<div class="card">

<div class="card-title">
    Packets
</div>

<div class="card-value">
    {_html_escape(
        extraction.get(
            "packet_count",
            "N/A"
        )
    )}
</div>

</div>


<div class="card">

<div class="card-title">
    Flows
</div>

<div class="card-value">
    {_html_escape(
        extraction.get(
            "flow_count",
            "N/A"
        )
    )}
</div>

</div>


<div class="card">

<div class="card-title">
    ML Features
</div>

<div class="card-value">
    {_html_escape(
        features.get(
            "feature_count",
            "N/A"
        )
    )}
</div>

</div>


<div class="card">

<div class="card-title">
    Attack Predictions
</div>

<div class="card-value">
    {_html_escape(
        predictions.get(
            "attack_count",
            "N/A"
        )
    )}
</div>

</div>


</div>


<h2>
    1. PCAP Information
</h2>

<table>

<tr>
    <th>Property</th>
    <th>Value</th>
</tr>

{_table_rows(pcap)}

</table>


<h2>
    2. Packet and Flow Extraction
</h2>

<table>

<tr>
    <th>Property</th>
    <th>Value</th>
</tr>

{_table_rows(extraction)}

</table>


<h2>
    3. Feature Engineering
</h2>

<table>

<tr>
    <th>Property</th>
    <th>Value</th>
</tr>

{_table_rows(features)}

</table>


<h2>
    4. TabPFN Model
</h2>

<table>

<tr>
    <th>Property</th>
    <th>Value</th>
</tr>

{_table_rows(model)}

</table>
"""

    if inference is not None:
        html += f"""

<h2>
    Inference Architecture &amp; Execution
</h2>

<table>

<tr>
    <th>Property</th>
    <th>Value</th>
</tr>

{_table_rows(inference)}

</table>
"""

    html += f"""

<h2>
    5. Prediction Results
</h2>

<table>

<tr>
    <th>Property</th>
    <th>Value</th>
</tr>

{_table_rows(predictions)}

</table>
"""

    if ground_truth is not None:

        html += f"""

<h2>
    6. Ground Truth Matching
</h2>

<table>

<tr>
    <th>Property</th>
    <th>Value</th>
</tr>

{_table_rows(ground_truth)}

</table>
"""

    if evaluation is not None:

        html += f"""

<h2>
    7. Evaluation Metrics
</h2>

<table>

<tr>
    <th>Metric</th>
    <th>Value</th>
</tr>

{_table_rows(evaluation)}

</table>
"""

    html += """

<div class="footer">

Generated by the
TabPFN-NIDS PCAP Analysis Pipeline.

</div>

</div>

</body>

</html>
"""

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(html)

    return output_path


def build_report_data(
    pcap_path,
    extraction_summary,
    features,
    feature_names,
    predictions,
    probabilities,
    total_analysis_seconds,
    ground_truth_path=None,
    metrics=None,
    labeled_flows=None,
    inference_meta=None,
):
    """Build structured data for JSON and HTML reports."""

    total_predictions = len(
        predictions
    )

    normal_count = int(
        (predictions == 0).sum()
    )

    attack_count = int(
        (predictions == 1).sum()
    )

    normal_percentage = (
        normal_count
        / total_predictions
        * 100
        if total_predictions
        else 0.0
    )

    attack_percentage = (
        attack_count
        / total_predictions
        * 100
        if total_predictions
        else 0.0
    )

    pcap_size_bytes = (
        pcap_path.stat().st_size
    )

    report_data = {

        "generated_at":
            datetime.now().isoformat(),

        "pcap": {

            "filename":
                pcap_path.name,

            "path":
                str(pcap_path),

            "size_bytes":
                pcap_size_bytes,

            "size_mb":
                round(
                    pcap_size_bytes
                    / (1024 * 1024),
                    4,
                ),
        },

        "extraction": {

            "packet_count":
                extraction_summary.get(
                    "total_packets",
                    0,
                ),

            "flow_count":
                extraction_summary.get(
                    "total_flows",
                    0,
                ),

            "backend":
                "scapy",

            "flow_timeout_seconds":
                120,

            "idle_timeout_seconds":
                60,
        },

        "features": {

            "flow_rows":
                len(features),

            "feature_count":
                len(feature_names),

            "feature_columns":
                list(feature_names),

            "total_columns_after_engineering":
                len(features.columns),
        },

        "model": {

            "model_type":
                "TabPFN",

            "task":
                "binary",

            "device":
                "CPU",

            "feature_count":
                len(feature_names),

            "samples_predicted":
                total_predictions,
        },

        "predictions": {

            "total":
                total_predictions,

            "normal_count":
                normal_count,

            "attack_count":
                attack_count,

            "normal_percentage":
                round(
                    normal_percentage,
                    2,
                ),

            "attack_percentage":
                round(
                    attack_percentage,
                    2,
                ),
        },

        "performance": {

            "total_analysis_seconds":
                round(
                    total_analysis_seconds,
                    4,
                ),
        },

        "ground_truth":
            None,

        "evaluation":
            None,

        "inference":
            None,
    }

    if inference_meta is not None:
        rows_per_chunk = inference_meta.get("rows_per_chunk", [])
        if isinstance(rows_per_chunk, list):
            rows_per_chunk_str = ", ".join(str(r) for r in rows_per_chunk)
        else:
            rows_per_chunk_str = str(rows_per_chunk)

        report_data["inference"] = {
            "Inference mode":
                inference_meta.get("inference_mode", "Single model"),
            "Number of input flows":
                inference_meta.get("num_flows", total_predictions),
            "Number of models":
                inference_meta.get("num_models", 1),
            "Number of workers":
                inference_meta.get("num_workers", 1),
            "Chunk size":
                inference_meta.get("max_rows_per_worker", 10000),
            "Number of chunks":
                inference_meta.get("num_chunks", 1),
            "Rows per chunk":
                rows_per_chunk_str,
            "Ensemble method":
                inference_meta.get("ensemble_method", "N/A"),
            "Threshold":
                inference_meta.get("prediction_threshold", 0.5),
            "Models used":
                ", ".join(inference_meta.get("models_used", ["tabpfn_binary_model"])),
            "Total inference time (s)":
                inference_meta.get("total_inference_seconds", 0.0),
            "Average model inference time (s)":
                inference_meta.get("avg_model_inference_seconds", 0.0),
        }

    if (
        ground_truth_path is not None
        and labeled_flows is not None
    ):

        matched_mask = (
            labeled_flows[
                "label"
            ] != -1
        )

        matched_count = int(
            matched_mask.sum()
        )

        unmatched_count = int(
            len(labeled_flows)
            - matched_count
        )

        report_data[
            "ground_truth"
        ] = {

            "file":
                str(
                    ground_truth_path
                ),

            "matched_flows":
                matched_count,

            "unmatched_flows":
                unmatched_count,

            "match_rate_percentage":
                round(
                    matched_count
                    / len(labeled_flows)
                    * 100,
                    2,
                )
                if len(labeled_flows)
                else 0.0,
        }

    if metrics is not None:

        report_data[
            "evaluation"
        ] = {

            "accuracy":
                metrics.get(
                    "accuracy"
                ),

            "precision":
                metrics.get(
                    "precision"
                ),

            "recall":
                metrics.get(
                    "recall"
                ),

            "f1_score":
                metrics.get(
                    "f1_score"
                ),

            "roc_auc":
                metrics.get(
                    "roc_auc"
                ),

            "true_negatives":
                metrics.get(
                    "true_negatives"
                ),

            "false_positives":
                metrics.get(
                    "false_positives"
                ),

            "false_negatives":
                metrics.get(
                    "false_negatives"
                ),

            "true_positives":
                metrics.get(
                    "true_positives"
                ),
        }

    return report_data

def save_docx_report(
    report_data: dict,
    output_path: Path,
) -> Path:
    """Save the PCAP analysis report as a Word document."""

    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.shared import Pt

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    document = Document()

    # -----------------------------------------------------
    # Title
    # -----------------------------------------------------

    title = document.add_heading(
        "PCAP Analysis Report",
        level=0,
    )

    title.alignment = (
        WD_ALIGN_PARAGRAPH.CENTER
    )

    pcap = report_data.get(
        "pcap",
        {},
    )

    extraction = report_data.get(
        "extraction",
        {},
    )

    features = report_data.get(
        "features",
        {},
    )

    model = report_data.get(
        "model",
        {},
    )

    predictions = report_data.get(
        "predictions",
        {},
    )

    performance = report_data.get(
        "performance",
        {},
    )

    ground_truth = report_data.get(
        "ground_truth"
    )

    evaluation = report_data.get(
        "evaluation"
    )

    inference = report_data.get(
        "inference"
    )

    generated_at = report_data.get(
        "generated_at",
        "N/A",
    )

    document.add_paragraph(
        f"Generated: {generated_at}"
    )

    # -----------------------------------------------------
    # Helper
    # -----------------------------------------------------

    def add_table(
        title,
        data,
    ):

        document.add_heading(
            title,
            level=1,
        )

        table = document.add_table(
            rows=1,
            cols=2,
        )

        table.alignment = (
            WD_TABLE_ALIGNMENT.CENTER
        )

        table.style = (
            "Table Grid"
        )

        header = table.rows[0].cells

        header[0].text = "Property"
        header[1].text = "Value"

        for key, value in data.items():

            row = table.add_row().cells

            row[0].text = str(key)

            if isinstance(
                value,
                float,
            ):
                value = f"{value:.4f}"

            row[1].text = str(value)

    # -----------------------------------------------------
    # 1. PCAP Information
    # -----------------------------------------------------

    add_table(
        "1. PCAP Information",
        pcap,
    )

    # -----------------------------------------------------
    # 2. Packet and Flow Extraction
    # -----------------------------------------------------

    add_table(
        "2. Packet and Flow Extraction",
        extraction,
    )

    # -----------------------------------------------------
    # 3. Feature Engineering
    # -----------------------------------------------------

    feature_summary = {
        "Flow rows":
            features.get(
                "flow_rows",
                "N/A",
            ),

        "ML feature count":
            features.get(
                "feature_count",
                "N/A",
            ),

        "Total columns after engineering":
            features.get(
                "total_columns_after_engineering",
                "N/A",
            ),
    }

    add_table(
        "3. Feature Engineering",
        feature_summary,
    )

    # -----------------------------------------------------
    # 4. Feature Schema
    # -----------------------------------------------------

    document.add_heading(
        "4. Model Feature Schema",
        level=1,
    )

    feature_columns = features.get(
        "feature_columns",
        [],
    )

    for index, feature in enumerate(
        feature_columns,
        start=1,
    ):

        document.add_paragraph(
            f"{index}. {feature}"
        )

    # -----------------------------------------------------
    # 5. TabPFN Model
    # -----------------------------------------------------

    add_table(
        "5. TabPFN Model",
        model,
    )

    if inference:
        add_table(
            "Inference Architecture & Execution",
            inference,
        )

    # -----------------------------------------------------
    # 6. Prediction Results
    # -----------------------------------------------------

    add_table(
        "6. Prediction Results",
        predictions,
    )

    # -----------------------------------------------------
    # 7. Performance
    # -----------------------------------------------------

    add_table(
        "7. Processing Performance",
        performance,
    )

    # -----------------------------------------------------
    # 8. Ground Truth
    # -----------------------------------------------------

    if ground_truth is not None:

        add_table(
            "8. Ground Truth Matching",
            ground_truth,
        )

    # -----------------------------------------------------
    # 9. Evaluation
    # -----------------------------------------------------

    if evaluation is not None:

        add_table(
            "9. Evaluation Metrics",
            evaluation,
        )

    # -----------------------------------------------------
    # 10. Final Summary
    # -----------------------------------------------------

    document.add_heading(
        "10. Analysis Summary",
        level=1,
    )

    total = predictions.get(
        "total",
        0,
    )

    normal = predictions.get(
        "normal_count",
        0,
    )

    attack = predictions.get(
        "attack_count",
        0,
    )

    document.add_paragraph(
        f"The analyzed PCAP contained "
        f"{total} network flows."
    )

    document.add_paragraph(
        f"The TabPFN-NIDS model classified "
        f"{normal} flows as Normal and "
        f"{attack} flows as Attack."
    )

    if evaluation is not None:

        document.add_paragraph(
            "Ground-truth evaluation was "
            "performed for the flows that "
            "successfully matched the supplied "
            "ground-truth dataset."
        )

    else:

        document.add_paragraph(
            "No ground-truth dataset was supplied "
            "for this analysis, so prediction "
            "metrics against ground truth were "
            "not calculated."
        )

    # -----------------------------------------------------
    # Footer
    # -----------------------------------------------------

    section = document.sections[0]

    footer = section.footer.paragraphs[0]

    footer.text = (
        "Generated by the TabPFN-NIDS "
        "PCAP Analysis Pipeline"
    )

    footer.alignment = (
        WD_ALIGN_PARAGRAPH.CENTER
    )

    # -----------------------------------------------------
    # Font
    # -----------------------------------------------------

    for paragraph in document.paragraphs:

        for run in paragraph.runs:

            run.font.name = "Arial"
            run.font.size = Pt(10)

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    document.save(
        output_path
    )

    return output_path