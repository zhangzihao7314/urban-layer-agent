"""Canonical, non-destructive schema for Layer Alterator Agent outputs."""

from datetime import datetime, timezone


SCHEMA_VERSION = "2.0"


def preserve_source_semantics(gdf):
    """Keep source ``transf_type`` and create an explicit immutable copy."""
    result = gdf.copy()
    if "transf_type" in result.columns and "original_transf_type" not in result.columns:
        result["original_transf_type"] = result["transf_type"]
    return result


def migrate_legacy_columns(gdf):
    result = preserve_source_semantics(gdf)
    rename = {}
    if "matched_urban_type" in result.columns and "target_lcz_type" not in result.columns:
        rename["matched_urban_type"] = "target_lcz_type"
    if "user_description" in result.columns and "user_request" not in result.columns:
        rename["user_description"] = "user_request"
    result = result.rename(columns=rename)
    result["schema_version"] = SCHEMA_VERSION
    return result


def build_lineage(source_path: str, reference_path: str = "") -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_vector": str(source_path),
        "reference_table": str(reference_path),
        "transformation": "Agent decision -> LCZ reference lookup -> Layer Alterator inputs",
    }
