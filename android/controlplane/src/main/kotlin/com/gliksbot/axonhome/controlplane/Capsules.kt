package com.gliksbot.axonhome.controlplane

import kotlinx.datetime.Instant
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
enum class CapsuleKind {
    /** Full organism continuation boundary (arch §9.2). */
    @SerialName("full_organism") FULL_ORGANISM,
    /** Parameter payload + record; restores weights, not the organism. */
    @SerialName("checkpoint") CHECKPOINT,
    /** A git revision; recoverable from GitHub alone. */
    @SerialName("code") CODE,
}

/**
 * Recovery capsule manifest. Deliberately compatible with the repo's detached
 * axon-cloud-bundle-manifest-v1 (runtime/trainer/cloud_bundle.py): per-member
 * {path: {sha256, bytes}} + archive digest, manifest written LAST (arch §9.2).
 */
@Serializable
data class CapsuleManifest(
    val schema: String = SCHEMA,
    @SerialName("capsule_id") val capsuleId: String,
    val kind: CapsuleKind? = null,
    @SerialName("archive_name") val archiveName: String? = null,
    @SerialName("archive_sha256") val archiveSha256: String,
    @SerialName("archive_bytes") val archiveBytes: Long,
    /** Archive path -> {sha256, bytes}. */
    val members: Map<String, ManifestMember>,
    @SerialName("architecture_id") val architectureId: String? = null,
    @SerialName("base_generation_id") val baseGenerationId: String? = null,
    @SerialName("candidate_generation_id") val candidateGenerationId: String? = null,
    @SerialName("checkpoint_id") val checkpointId: String? = null,
    @SerialName("soul_ids") val soulIds: List<String> = emptyList(),
    /** HEAD sha at export. */
    @SerialName("git_commit") val gitCommit: String,
    @SerialName("created_at") val createdAt: Instant,
    val storage: CapsuleStorage? = null,
) {
    companion object {
        const val SCHEMA = "axon-cloud-bundle-manifest-v1"
    }
}

@Serializable
data class ManifestMember(
    val sha256: String,
    val bytes: Long,
)

@Serializable
data class CapsuleStorage(
    val provider: String,
    val path: String,
)

@Serializable
enum class CapsuleVerificationStatus {
    @SerialName("unverified") UNVERIFIED,
    @SerialName("verified") VERIFIED,
    @SerialName("quarantined") QUARANTINED,
}

@Serializable
data class CapsuleSummary(
    @SerialName("capsule_id") val capsuleId: String,
    val kind: CapsuleKind,
    @SerialName("archive_sha256") val archiveSha256: String,
    @SerialName("archive_bytes") val archiveBytes: Long? = null,
    @SerialName("created_at") val createdAt: Instant,
    @SerialName("verification_status") val verificationStatus: CapsuleVerificationStatus,
)

/** GET /v1/capsules. */
@Serializable
data class CapsuleList(
    val schema: String = SCHEMA,
    val freshness: Freshness,
    @SerialName("observed_at") val observedAt: Instant,
    val capsules: List<CapsuleSummary>,
) {
    companion object {
        const val SCHEMA = "axon-controlplane-capsule-list-v1"
    }
}

/** POST /v1/capsules — export request. */
@Serializable
data class CapsuleExportRequest(
    val schema: String = SCHEMA,
    val kind: CapsuleKind,
    @SerialName("module_id") val moduleId: String? = null,
    @SerialName("candidate_generation_id") val candidateGenerationId: String? = null,
    @SerialName("storage_provider") val storageProvider: String? = null,
) {
    companion object {
        const val SCHEMA = "axon-controlplane-capsule-export-v1"
    }
}

@Serializable
enum class CapsuleCheckName {
    @SerialName("archive_sha256") ARCHIVE_SHA256,
    @SerialName("member_set") MEMBER_SET,
    @SerialName("member_sha256") MEMBER_SHA256,
    @SerialName("architecture_identity") ARCHITECTURE_IDENTITY,
    @SerialName("candidate_generation") CANDIDATE_GENERATION,
    @SerialName("optimizer_policy") OPTIMIZER_POLICY,
    @SerialName("soul_checkpoint_pairing") SOUL_CHECKPOINT_PAIRING,
    @SerialName("lineage") LINEAGE,
}

@Serializable
data class CapsuleCheck(
    val name: CapsuleCheckName,
    val ok: Boolean,
    val detail: String? = null,
)

/**
 * Restore-verification report (POST /v1/capsules/{id}/verify), mirroring
 * axon-cloud-bundle-verification-v1 discipline (arch §9.3): fail-closed —
 * any mismatch quarantines the whole capsule; nothing half-verified reaches
 * canonical State.
 */
@Serializable
data class CapsuleVerificationReport(
    val schema: String = SCHEMA,
    @SerialName("capsule_id") val capsuleId: String,
    val ok: Boolean,
    val checks: List<CapsuleCheck>,
    val mismatches: List<String> = emptyList(),
    /** Set when ok=false: all members routed to quarantine. */
    @SerialName("quarantine_dir") val quarantineDir: String? = null,
    @SerialName("verified_at") val verifiedAt: Instant,
) {
    companion object {
        const val SCHEMA = "axon-cloud-bundle-verification-v1"
    }
}
