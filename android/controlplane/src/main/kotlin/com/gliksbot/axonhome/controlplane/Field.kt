package com.gliksbot.axonhome.controlplane

import kotlinx.datetime.Instant
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/** GET /v1/field/head — active branch HEAD + health. Read-only; the Control Plane never writes canonical field state. */
@Serializable
data class FieldHead(
    val schema: String = SCHEMA,
    val freshness: Freshness,
    @SerialName("observed_at") val observedAt: Instant,
    val head: BranchHead,
    val health: HeartHealthSummary,
) {
    companion object {
        const val SCHEMA = "axon-controlplane-field-head-v1"
    }
}

/** Distilled active/branches/active/HEAD.json (axon-canonical-state-branch-head-v1). */
@Serializable
data class BranchHead(
    val schema: String = SCHEMA,
    val generation: Long,
    @SerialName("field_id") val fieldId: String,
    @SerialName("tick_id") val tickId: Long,
    @SerialName("parent_field_id") val parentFieldId: String? = null,
) {
    companion object {
        const val SCHEMA = "axon-canonical-state-branch-head-v1"
    }
}

/** GET /v1/field/snapshot/{fieldId} — reference view over a shared-field-v4 snapshot. */
@Serializable
data class FieldSnapshotView(
    val schema: String = SCHEMA,
    val freshness: Freshness,
    @SerialName("observed_at") val observedAt: Instant,
    /** The full snapshot carries schema shared-field-v4. */
    @SerialName("snapshot_schema") val snapshotSchema: String = SNAPSHOT_SCHEMA,
    @SerialName("field_id") val fieldId: String,
    @SerialName("tick_id") val tickId: Long,
    @SerialName("parent_field_id") val parentFieldId: String? = null,
    val regions: List<RegionSummary>,
    @SerialName("canonical_sha256") val canonicalSha256: String,
) {
    companion object {
        const val SCHEMA = "axon-controlplane-field-snapshot-ref-v1"
        const val SNAPSHOT_SCHEMA = "shared-field-v4"
    }
}

@Serializable
data class RegionSummary(
    val region: LogicalRegion,
    @SerialName("span_count") val spanCount: Long,
    @SerialName("char_count") val charCount: Long,
    val visibility: RegionVisibility,
    @SerialName("write_policy") val writePolicy: RegionWritePolicy,
    @SerialName("region_sha256") val regionSha256: String,
)

@Serializable
enum class RegionVisibility {
    @SerialName("attended") ATTENDED,
    @SerialName("masked") MASKED,
}

@Serializable
enum class RegionWritePolicy {
    @SerialName("sealed") SEALED,
    @SerialName("core_writable") CORE_WRITABLE,
}

/** GET /v1/field/ticks/{n} — tail of journal.jsonl with delta refs. */
@Serializable
data class TickList(
    val schema: String = SCHEMA,
    val freshness: Freshness,
    @SerialName("observed_at") val observedAt: Instant,
    val ticks: List<TickRecord>,
) {
    companion object {
        const val SCHEMA = "axon-controlplane-tick-list-v1"
    }
}

/** One journal entry (axon-canonical-state-branch-event-v1). */
@Serializable
data class TickRecord(
    val schema: String = SCHEMA,
    val event: BranchEventKind,
    val generation: Long,
    @SerialName("tick_id") val tickId: Long,
    @SerialName("base_field_id") val baseFieldId: String? = null,
    @SerialName("successor_field_id") val successorFieldId: String,
    @SerialName("delta_ids") val deltaIds: List<String> = emptyList(),
    @SerialName("heart_commit_id") val heartCommitId: String? = null,
    @SerialName("consolidator_core_id") val consolidatorCoreId: String? = null,
    val timestamp: Instant? = null,
) {
    companion object {
        const val SCHEMA = "axon-canonical-state-branch-event-v1"
    }
}

@Serializable
enum class BranchEventKind {
    @SerialName("commit") COMMIT,
    @SerialName("initialize") INITIALIZE,
    @SerialName("schema_migration") SCHEMA_MIGRATION,
}
