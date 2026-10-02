"""
Fixed vocabularies the LLM must choose from. Keeping these as plain Python
constants (not hardcoded inside prompts) means you change them in one place
and every prompt that uses them picks up the change automatically.
"""

ISSUE_TAXONOMY = {
    "Product": ["Size/Fit", "Quality", "Features", "Product Information"],
    "Delivery": ["Late Delivery", "Damaged Package", "Tracking"],
    "Refund": ["Refund Delay", "Incorrect Refund"],
    "Service": ["Customer Service", "Communication"],
}

SENTIMENT = ["Positive", "Neutral", "Negative", "Mixed"]
SEVERITY = ["Low", "Medium", "High", "Critical"]
DECISIONS = ["INVESTIGATE", "MONITOR", "INSUFFICIENT_DATA"]
VERDICTS = ["SUPPORTED", "MIXED", "INCONCLUSIVE"]
