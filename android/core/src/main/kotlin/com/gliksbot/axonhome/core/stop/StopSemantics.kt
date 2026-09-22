package com.gliksbot.axonhome.core.stop

import kotlinx.serialization.Serializable

/**
 * Stop semantics (mission §13). Two kinds only:
 *  - SAFE: stop at the next atomic Trainer/runtime boundary and preserve
 *    recoverability; reports exactly what was preserved.
 *  - EMERGENCY: terminate immediately; reports a per-component ack map. The
 *    overall result may NOT claim all-stopped unless every registered
 *    component acknowledged (or is provably gone — never faked).
 */

enum class StopKind {
    SAFE,
    EMERGENCY,
}

@Serializable
enum class ComponentAck {
    @kotlinx.serialization.SerialName("stopped") STOPPED,
    @kotlinx.serialization.SerialName("unreachable") UNREACHABLE,
    @kotlinx.serialization.SerialName("failed") FAILED,
}

/** Result of a safe stop at an atomic boundary. */
@Serializable
data class SafeStopResult(
    val kind: String = "safe",
    /** The atomic boundary at which execution actually stopped. */
    val boundary: String,
    /** Identifiers of everything preserved for recovery (checkpoints, souls, pointers). */
    val preserved: List<String>,
    val recoverable: Boolean = true,
)

/** Result of an emergency stop: exact per-component acknowledgements. */
@Serializable
data class EmergencyStopResult(
    val kind: String = "emergency",
    /** componentId → acknowledgement. Never claims success for silent components. */
    val acknowledgements: Map<String, ComponentAck>,
) {
    /** True ONLY if every registered component acknowledged STOPPED. */
    val allStopped: Boolean
        get() = acknowledgements.isNotEmpty() && acknowledgements.values.all { it == ComponentAck.STOPPED }

    /** Human-legible summary; explicitly lists anything not confirmed stopped. */
    val summary: String
        get() = if (allStopped) {
            "all ${acknowledgements.size} components confirmed stopped"
        } else {
            val notStopped = acknowledgements.filterValues { it != ComponentAck.STOPPED }
            "NOT fully stopped: " + notStopped.entries.joinToString(", ") {
                "${it.key}=${it.value.name.lowercase()}"
            }
        }
}

sealed class StopResult {
    data class Safe(val result: SafeStopResult) : StopResult()
    data class Emergency(val result: EmergencyStopResult) : StopResult()
}
