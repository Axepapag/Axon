package com.gliksbot.axonhome.controlplane

import kotlinx.datetime.Instant
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Capability advertisement per arch §7.1 (ComputeProvider / NotebookProvider).
 * The UI renders controls ONLY from advertised capabilities; a provider that
 * cannot do something simply does not show the control.
 */
@Serializable
data class CapabilitySet(
    val supportsLaunch: Boolean = false,
    val supportsStop: Boolean = false,
    val supportsStatus: Boolean = false,
    val supportsLogs: Boolean = false,
    val supportsGpuMetrics: Boolean = false,
    val supportsProvision: Boolean = false,
    val supportsDestroy: Boolean = false,
    val supportsFiles: Boolean = false,
    val supportsTerminal: Boolean = false,
    val supportsArtifacts: Boolean = false,
    // StorageProvider
    val supportsBrowse: Boolean = false,
    val supportsUpload: Boolean = false,
    val supportsDownload: Boolean = false,
    val supportsHashVerify: Boolean = false,
    // AgentProvider
    val supportsStreaming: Boolean = false,
    val supportsTools: Boolean = false,
    val supportsSystemPrompt: Boolean = false,
    val supportsModelList: Boolean = false,
    // GitProvider
    val supportsPush: Boolean = false,
    val supportsPullRequest: Boolean = false,
    val supportsDiff: Boolean = false,
    val supportsCommit: Boolean = false,
    // AxonRuntimeProvider
    val supportsPause: Boolean = false,
    val supportsMaskControl: Boolean = false,
    val supportsIngress: Boolean = false,
    val supportsLiveStream: Boolean = false,
) {
    /** All advertised capability names that are enabled (no reflection — kotlin-reflect is not a dependency). */
    fun advertised(): Set<String> = buildSet {
        if (supportsLaunch) add("supportsLaunch")
        if (supportsStop) add("supportsStop")
        if (supportsStatus) add("supportsStatus")
        if (supportsLogs) add("supportsLogs")
        if (supportsGpuMetrics) add("supportsGpuMetrics")
        if (supportsProvision) add("supportsProvision")
        if (supportsDestroy) add("supportsDestroy")
        if (supportsFiles) add("supportsFiles")
        if (supportsTerminal) add("supportsTerminal")
        if (supportsArtifacts) add("supportsArtifacts")
        if (supportsBrowse) add("supportsBrowse")
        if (supportsUpload) add("supportsUpload")
        if (supportsDownload) add("supportsDownload")
        if (supportsHashVerify) add("supportsHashVerify")
        if (supportsStreaming) add("supportsStreaming")
        if (supportsTools) add("supportsTools")
        if (supportsSystemPrompt) add("supportsSystemPrompt")
        if (supportsModelList) add("supportsModelList")
        if (supportsPush) add("supportsPush")
        if (supportsPullRequest) add("supportsPullRequest")
        if (supportsDiff) add("supportsDiff")
        if (supportsCommit) add("supportsCommit")
        if (supportsPause) add("supportsPause")
        if (supportsMaskControl) add("supportsMaskControl")
        if (supportsIngress) add("supportsIngress")
        if (supportsLiveStream) add("supportsLiveStream")
    }
}

@Serializable
enum class WorkerStatus {
    @SerialName("registered") REGISTERED,
    @SerialName("connected") CONNECTED,
    @SerialName("busy") BUSY,
    @SerialName("lost") LOST,
    @SerialName("stopped") STOPPED,
    @SerialName("destroyed") DESTROYED,
}

/** Machine registry record (arch §5.5). Operational (non-canonical) state. */
@Serializable
data class Worker(
    @SerialName("worker_id") val workerId: String,
    /** e.g. ssh, kaggle, colab, jupyter, vm, local. */
    val provider: String,
    val endpoint: String,
    val label: String? = null,
    val status: WorkerStatus,
    val capabilities: CapabilitySet = CapabilitySet(),
    /** e.g. "nvidia-t4", "tpu-v2"; null for CPU-only. */
    val accelerator: String? = null,
    @SerialName("last_heartbeat_at") val lastHeartbeatAt: Instant? = null,
    @SerialName("active_job_id") val activeJobId: String? = null,
    @SerialName("worker_version") val workerVersion: String? = null,
    @SerialName("git_commit") val gitCommit: String? = null,
    @SerialName("registered_at") val registeredAt: Instant,
)

/** GET /v1/compute/workers. */
@Serializable
data class WorkerList(
    val schema: String = SCHEMA,
    val freshness: Freshness,
    @SerialName("observed_at") val observedAt: Instant,
    val workers: List<Worker>,
) {
    companion object {
        const val SCHEMA = "axon-controlplane-worker-list-v1"
    }
}

/** POST /v1/compute/workers. */
@Serializable
data class RegisterWorkerRequest(
    val schema: String = SCHEMA,
    val provider: String,
    val endpoint: String,
    val label: String? = null,
    val capabilities: CapabilitySet = CapabilitySet(),
    val accelerator: String? = null,
) {
    companion object {
        const val SCHEMA = "axon-controlplane-worker-register-v1"
    }
}

/** POST /v1/compute/workers/{workerId}/stop. */
@Serializable
data class WorkerStopResponse(
    val schema: String = SCHEMA,
    val freshness: Freshness,
    @SerialName("observed_at") val observedAt: Instant,
    @SerialName("worker_id") val workerId: String,
    val ack: ComponentAck,
) {
    companion object {
        const val SCHEMA = "axon-controlplane-worker-stop-v1"
    }
}
