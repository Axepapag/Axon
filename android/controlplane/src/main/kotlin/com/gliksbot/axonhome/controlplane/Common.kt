package com.gliksbot.axonhome.controlplane

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Data freshness labels (arch §4.3/§12). Every stateful Control Plane response
 * carries one. Cached state is never presented as current. SIMULATION is a
 * UI-side fifth state added by the app's simulator; it is never produced by
 * this API and so is deliberately not part of [Freshness].
 */
@Serializable
enum class Freshness {
    LIVE,
    CACHED,
    STALE,
    DISCONNECTED,
}

/** Typed error codes (schemas/common.json). */
@Serializable
enum class ErrorCode {
    UNAUTHENTICATED,
    FORBIDDEN,
    NOT_FOUND,
    CONFLICT,
    TRAINER_COMMAND_UNAVAILABLE,
    TRAINER_COMMAND_REJECTED,
    STALE_DELTA,
    SEALED_REGION,
    MASK_CONFIRMATION_REQUIRED,
    IDEMPOTENCY_REPLAY,
    UNSUPPORTED_SCHEMA_VERSION,
    SEQ_GAP,
    COMPONENT_UNREACHABLE,
    VALIDATION_FAILED,
    QUARANTINED,
    UPSTREAM_ERROR,
    INTERNAL,
}

/**
 * Error envelope (arch §5.6):
 * `{schema: "axon-controlplane-error-v1", error: {code, message, retryable, upstream_status?}}`.
 * Governed failures pass through verbatim via [ApiError.upstreamStatus].
 */
@Serializable
data class ApiErrorEnvelope(
    val schema: String = ERROR_SCHEMA,
    val error: ApiError,
) {
    companion object {
        const val ERROR_SCHEMA = "axon-controlplane-error-v1"
    }
}

@Serializable
data class ApiError(
    val code: ErrorCode,
    val message: String,
    val retryable: Boolean,
    /** Governed failure passthrough, verbatim (e.g. Trainer "UNAVAILABLE"). */
    @SerialName("upstream_status") val upstreamStatus: String? = null,
    /** Present with UNSUPPORTED_SCHEMA_VERSION (HTTP 426). */
    @SerialName("supported_versions") val supportedVersions: List<String>? = null,
)

/** Canonical logical region ids, exact order (runtime/field/schema.py shared-field-v4). */
@Serializable
enum class LogicalRegion {
    @SerialName("conversation_history") CONVERSATION_HISTORY,
    @SerialName("user_input") USER_INPUT,
    @SerialName("cortex") CORTEX,
    @SerialName("situation_awareness") SITUATION_AWARENESS,
    @SerialName("tool_results") TOOL_RESULTS,
    @SerialName("advisor_input") ADVISOR_INPUT,
    @SerialName("task_state") TASK_STATE,
    @SerialName("scratch") SCRATCH,
    @SerialName("response_draft") RESPONSE_DRAFT,
    @SerialName("diary") DIARY,
    @SerialName("identity") IDENTITY,
    @SerialName("trainer_instructions") TRAINER_INSTRUCTIONS,
    @SerialName("training_responses") TRAINING_RESPONSES,
}
