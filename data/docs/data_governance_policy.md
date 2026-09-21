# Data Governance Policy

## Data Quality Standards
All operational datasets must meet the following minimum standards before being used for
management reporting:
- No duplicate transaction records.
- All dates stored in ISO 8601 format (YYYY-MM-DD).
- All monetary values stored as numeric values in BDT, without currency symbols.
- Missing critical fields (quantity, unit price) must be flagged, not silently imputed.

## Anomaly Escalation
Any transaction whose value deviates more than three standard deviations from the
category mean must be flagged for manual review before inclusion in reporting.

## Retention
Transactional procurement data is retained for seven years. Derived analytical datasets
are retained for three years.

## Access
Access to raw supplier pricing data is restricted to the Procurement and Finance
functions. AI systems querying this data must operate on read-only credentials.
