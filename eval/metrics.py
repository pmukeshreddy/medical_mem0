"""
Evaluation metrics for retrieval benchmarks.

Metrics:
- Recall@K: % of expected keywords found in top K results
- MRR: Mean Reciprocal Rank
- Keyword Precision: % of retrieved content matching expected
- Latency: p50, p95, p99
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
import statistics


@dataclass
class RetrievalMetrics:
    """Metrics for a single retrieval."""
    recall_at_3: float
    recall_at_5: float
    keyword_precision: float
    mrr: float
    latency_ms: float
    keywords_found: List[str]
    keywords_missing: List[str]


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across all evaluations."""
    strategy: str
    num_cases: int
    
    # Recall
    mean_recall_at_3: float
    mean_recall_at_5: float
    
    # Precision
    mean_keyword_precision: float
    
    # Ranking
    mean_mrr: float
    
    # Latency
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    
    # By difficulty
    easy_recall: float
    medium_recall: float
    hard_recall: float
    
    # By category
    category_recall: Dict[str, float]


def compute_keyword_recall(
    retrieved_contents: List[str], 
    expected_keywords: List[str],
    k: int = None
) -> Tuple[float, List[str], List[str]]:
    """
    Compute keyword-based recall.
    
    Args:
        retrieved_contents: List of retrieved memory contents
        expected_keywords: Keywords expected to be found
        k: Only consider top k results (None = all)
    
    Returns:
        (recall_score, found_keywords, missing_keywords)
    """
    if not expected_keywords:
        return 1.0, [], []
    
    # Combine retrieved contents
    if k:
        retrieved_contents = retrieved_contents[:k]
    
    combined = " ".join(retrieved_contents).lower()
    
    # Check each keyword
    found = []
    missing = []
    
    for keyword in expected_keywords:
        if keyword.lower() in combined:
            found.append(keyword)
        else:
            missing.append(keyword)
    
    recall = len(found) / len(expected_keywords)
    
    return recall, found, missing


def compute_mrr(
    retrieved_contents: List[str],
    expected_keywords: List[str]
) -> float:
    """
    Compute Mean Reciprocal Rank.
    
    MRR = 1/rank of first relevant result
    Relevant = contains at least one expected keyword
    """
    if not expected_keywords or not retrieved_contents:
        return 0.0
    
    for i, content in enumerate(retrieved_contents):
        content_lower = content.lower()
        for keyword in expected_keywords:
            if keyword.lower() in content_lower:
                return 1.0 / (i + 1)
    
    return 0.0


def compute_keyword_precision(
    retrieved_contents: List[str],
    expected_keywords: List[str]
) -> float:
    """
    Compute keyword precision.
    
    What % of retrieved results contain at least one expected keyword?
    """
    if not retrieved_contents or not expected_keywords:
        return 0.0
    
    relevant_count = 0
    
    for content in retrieved_contents:
        content_lower = content.lower()
        if any(kw.lower() in content_lower for kw in expected_keywords):
            relevant_count += 1
    
    return relevant_count / len(retrieved_contents)


def evaluate_single(
    retrieved_contents: List[str],
    expected_keywords: List[str],
    latency_ms: float
) -> RetrievalMetrics:
    """Evaluate a single retrieval."""
    
    recall_3, found_3, missing_3 = compute_keyword_recall(
        retrieved_contents, expected_keywords, k=3
    )
    
    recall_5, found_5, missing_5 = compute_keyword_recall(
        retrieved_contents, expected_keywords, k=5
    )
    
    mrr = compute_mrr(retrieved_contents, expected_keywords)
    precision = compute_keyword_precision(retrieved_contents, expected_keywords)
    
    return RetrievalMetrics(
        recall_at_3=recall_3,
        recall_at_5=recall_5,
        keyword_precision=precision,
        mrr=mrr,
        latency_ms=latency_ms,
        keywords_found=found_5,
        keywords_missing=missing_5
    )


def aggregate_metrics(
    results: List[Dict],  # [{case, metrics}, ...]
    strategy: str
) -> AggregatedMetrics:
    """Aggregate metrics across all evaluation cases."""
    
    if not results:
        return None
    
    # Extract metrics
    recall_3_scores = [r["metrics"].recall_at_3 for r in results]
    recall_5_scores = [r["metrics"].recall_at_5 for r in results]
    precision_scores = [r["metrics"].keyword_precision for r in results]
    mrr_scores = [r["metrics"].mrr for r in results]
    latencies = [r["metrics"].latency_ms for r in results]
    
    # By difficulty
    easy = [r["metrics"].recall_at_5 for r in results if r["case"].get("difficulty") == "easy"]
    medium = [r["metrics"].recall_at_5 for r in results if r["case"].get("difficulty") == "medium"]
    hard = [r["metrics"].recall_at_5 for r in results if r["case"].get("difficulty") == "hard"]
    
    # By category
    categories = {}
    for r in results:
        cat = r["case"].get("category", "other")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r["metrics"].recall_at_5)
    
    category_recall = {
        cat: statistics.mean(scores) if scores else 0.0
        for cat, scores in categories.items()
    }
    
    # Compute percentiles for latency
    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)
    
    return AggregatedMetrics(
        strategy=strategy,
        num_cases=len(results),
        mean_recall_at_3=statistics.mean(recall_3_scores),
        mean_recall_at_5=statistics.mean(recall_5_scores),
        mean_keyword_precision=statistics.mean(precision_scores),
        mean_mrr=statistics.mean(mrr_scores),
        mean_latency_ms=statistics.mean(latencies),
        p50_latency_ms=sorted_latencies[int(n * 0.50)] if n else 0,
        p95_latency_ms=sorted_latencies[int(n * 0.95)] if n else 0,
        p99_latency_ms=sorted_latencies[int(n * 0.99)] if n else 0,
        easy_recall=statistics.mean(easy) if easy else 0.0,
        medium_recall=statistics.mean(medium) if medium else 0.0,
        hard_recall=statistics.mean(hard) if hard else 0.0,
        category_recall=category_recall
    )


def format_metrics_table(metrics_list: List[AggregatedMetrics]) -> str:
    """Format metrics as ASCII table."""
    
    header = f"{'Strategy':<20} {'Recall@3':<10} {'Recall@5':<10} {'MRR':<10} {'Latency':<12} {'P95':<10}"
    separator = "-" * 72
    
    rows = [header, separator]
    
    # Sort by recall@5
    sorted_metrics = sorted(metrics_list, key=lambda x: x.mean_recall_at_5, reverse=True)
    
    for m in sorted_metrics:
        row = f"{m.strategy:<20} {m.mean_recall_at_3:<10.3f} {m.mean_recall_at_5:<10.3f} {m.mean_mrr:<10.3f} {m.mean_latency_ms:<12.1f} {m.p95_latency_ms:<10.1f}"
        rows.append(row)
    
    rows.append(separator)
    
    return "\n".join(rows)


def format_detailed_report(metrics: AggregatedMetrics) -> str:
    """Format detailed report for a single strategy."""
    
    lines = [
        f"=== {metrics.strategy.upper()} ===",
        "",
        "Overall Metrics:",
        f"  Recall@3: {metrics.mean_recall_at_3:.3f}",
        f"  Recall@5: {metrics.mean_recall_at_5:.3f}",
        f"  MRR: {metrics.mean_mrr:.3f}",
        f"  Precision: {metrics.mean_keyword_precision:.3f}",
        "",
        "Latency:",
        f"  Mean: {metrics.mean_latency_ms:.1f}ms",
        f"  P50: {metrics.p50_latency_ms:.1f}ms",
        f"  P95: {metrics.p95_latency_ms:.1f}ms",
        f"  P99: {metrics.p99_latency_ms:.1f}ms",
        "",
        "By Difficulty:",
        f"  Easy: {metrics.easy_recall:.3f}",
        f"  Medium: {metrics.medium_recall:.3f}",
        f"  Hard: {metrics.hard_recall:.3f}",
        "",
        "By Category:",
    ]
    
    for cat, score in sorted(metrics.category_recall.items()):
        lines.append(f"  {cat}: {score:.3f}")
    
    return "\n".join(lines)
