package com.gliksbot.axonhome.controlplane

import kotlinx.datetime.Instant
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
enum class ArtifactKind {
    @SerialName("checkpoint") CHECKPOINT,
    @SerialName("capsule") CAPSULE,
    @SerialName("cloud_bundle") CLOUD_BUNDLE,
    @SerialName("dormant_archive") DORMANT_ARCHIVE,
    @SerialName("report") REPORT,
    @SerialName("log") LOG,
    @SerialName("other") OTHER,
}

/**
 * Storage artifact catalog entry (arch §5.5/§12). Metadata mirror only: large
 * artifacts stay remote and content-addressed; the catalog carries hashes,
 * never bulk payloads.
 */
@Serializable
data class Artifact(
    @SerialName("artifact_id") val artifactId: String,
    val kind: ArtifactKind,
    /** Storage provider id: drive, sftp, s3, gcs, local, http. */
    val provider: String,
    val path: String,
    val sha256: String,
    val bytes: Long,
    @SerialName("created_at") val createdAt: Instant,
    /** True when path derives from content hash (e.g. checkpoints/<sha256>.pt). */
    @SerialName("content_addressed") val contentAddressed: Boolean = false,
    /** Last successful hash verification; null = never verified. */
    @SerialName("verified_at") val verifiedAt: Instant? = null,
)

/** GET /v1/storage/artifacts. */
@Serializable
data class ArtifactList(
    val schema: String = SCHEMA,
    val freshness: Freshness,
    @SerialName("observed_at") val observedAt: Instant,
    val artifacts: List<Artifact>,
) {
    companion object {
        const val SCHEMA = "axon-controlplane-artifact-list-v1"
    }
}

/** POST /v1/storage/artifacts. */
@Serializable
data class RegisterArtifactRequest(
    val schema: String = SCHEMA,
    val kind: ArtifactKind,
    val provider: String,
    val path: String,
    val sha256: String,
    val bytes: Long,
    @SerialName("content_addressed") val contentAddressed: Boolean = false,
) {
    companion object {
        const val SCHEMA = "axon-controlplane-artifact-register-v1"
    }
}
