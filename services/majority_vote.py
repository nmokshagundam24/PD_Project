"""
Majority Vote Module
Takes 6 task-level binary labels and computes the final prediction.
"""


def compute_majority_vote(labels: list[int]) -> tuple[int, dict]:
    """
    Compute majority vote across task-level predictions.
    
    Args:
        labels: List of binary labels (0 or 1), one per task.
    
    Returns:
        (final_label, breakdown) where:
          - final_label: 0 (Healthy) or 1 (Parkinson's)
          - breakdown: dict with vote counts
    """
    pd_count = labels.count(1)
    hc_count = labels.count(0)

    final_label = 1 if pd_count > hc_count else 0

    breakdown = {
        "parkinsons_votes": pd_count,
        "healthy_votes": hc_count,
        "total_tasks": len(labels),
    }

    return final_label, breakdown
