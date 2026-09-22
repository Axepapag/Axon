package com.gliksbot.axonhome.ui.state

import com.gliksbot.axonhome.controlplane.CapsuleCheck
import com.gliksbot.axonhome.controlplane.CapsuleVerificationReport
import com.gliksbot.axonhome.controlplane.ComponentAck
import com.gliksbot.axonhome.controlplane.ComponentAckStatus
import com.gliksbot.axonhome.controlplane.Freshness
import com.gliksbot.axonhome.controlplane.MaskPolicy
import com.gliksbot.axonhome.controlplane.MaskState
import com.gliksbot.axonhome.controlplane.StopAccounting
import com.gliksbot.axonhome.controlplane.StopResponse

/**
 * Pure screen-state mapping helpers. These functions are deliberately free of
 * Android imports so the JVM unit tests exercise exactly the logic the
 * composables render.
 */

/** Human label for a freshness state (arch §12). SIMULATION is banner-level. */
fun freshnessLabel(freshness: Freshness, simulation: Boolean): String =
    if (simulation) "SIMULATION · ${freshness.name}" else freshness.name

/** One rendered row of a stop result (arch §13 honesty rules). */
data class StopAckRow(
    val componentId: String,
    val kind: String,
    val status: ComponentAckStatus,
    val statusLabel: String,
    val detail: String?,
    val blocksAllStopped: Boolean,
)

/** Overall stop summary; never claims all-stopped unless the ack map proves it. */
data class StopSummary(
    val scopeLabel: String,
    val allStopped: Boolean,
    val claimVerified: Boolean,
    val headline: String,
    val rows: List<StopAckRow>,
)

fun ComponentAckStatus.label(): String = when (this) {
    ComponentAckStatus.CONFIRMED_STOPPED -> "confirmed stopped"
    ComponentAckStatus.UNREACHABLE -> "unreachable"
    ComponentAckStatus.PROVABLY_GONE -> "provably gone"
    ComponentAckStatus.REFUSED -> "refused"
    ComponentAckStatus.UNAVAILABLE -> "unavailable"
}

/**
 * Render a stop response truthfully: re-derive allStopped with
 * [StopAccounting] and flag any mismatch between the server's claim and the
 * ack map (arch §13: the app never displays "everything stopped" unless every
 * component acknowledged or is provably gone).
 */
fun summarizeStop(response: StopResponse): StopSummary {
    val recomputed = StopAccounting.computeAllStopped(response.components)
    val verified = StopAccounting.verify(response)
    val rows = response.components.map { ack: ComponentAck ->
        StopAckRow(
            componentId = ack.componentId,
            kind = ack.componentKind.name.lowercase().replace('_', ' '),
            status = ack.status,
            statusLabel = ack.status.label(),
            detail = ack.detail,
            blocksAllStopped = ack.status != ComponentAckStatus.CONFIRMED_STOPPED &&
                ack.status != ComponentAckStatus.PROVABLY_GONE,
        )
    }
    val headline = when {
        !verified -> "WARNING: response claim disagrees with acknowledgements — showing derived result"
        recomputed -> "All components confirmed stopped or provably gone"
        else -> "NOT everything is stopped — see components below"
    }
    return StopSummary(
        scopeLabel = response.scope.name.lowercase().replace('_', ' '),
        allStopped = recomputed,
        claimVerified = verified,
        headline = headline,
        rows = rows,
    )
}

/** One rendered row of a capsule verification report (arch §9.3). */
data class CapsuleCheckRow(
    val name: String,
    val ok: Boolean,
    val detail: String?,
)

data class CapsuleVerifySummary(
    val capsuleId: String,
    val ok: Boolean,
    val headline: String,
    val rows: List<CapsuleCheckRow>,
    val mismatches: List<String>,
    val quarantined: Boolean,
)

fun summarizeCapsuleReport(report: CapsuleVerificationReport): CapsuleVerifySummary =
    CapsuleVerifySummary(
        capsuleId = report.capsuleId,
        ok = report.ok,
        headline = if (report.ok) {
            "Capsule verified — all checks passed"
        } else {
            "Capsule FAILED verification — quarantined, nothing restored"
        },
        rows = report.checks.map { check: CapsuleCheck ->
            CapsuleCheckRow(
                name = check.name.name.lowercase().replace('_', ' '),
                ok = check.ok,
                detail = check.detail,
            )
        },
        mismatches = report.mismatches,
        quarantined = !report.ok && report.quarantineDir != null,
    )

/** Describe a mask policy for the two-step confirm dialog. */
fun maskPolicyLabel(policy: MaskPolicy): String = when {
    policy.kind == com.gliksbot.axonhome.controlplane.MaskKind.ALL -> "all"
    policy.kind == com.gliksbot.axonhome.controlplane.MaskKind.NONE -> "none"
    policy.kind == com.gliksbot.axonhome.controlplane.MaskKind.LAST_N_SPANS -> "last ${policy.n ?: 0} spans"
    else -> "newest ${policy.tailPercent ?: 0}%"
}

/** Diff two mask states into human-readable change rows for confirm dialogs. */
fun maskChanges(before: MaskState, after: MaskState): List<String> =
    after.policies.mapNotNull { (region, policy) ->
        val old = before.policies[region]
        if (old == policy) null
        else "${region.name.lowercase()}: ${maskPolicyLabel(old ?: policy)} → ${maskPolicyLabel(policy)}"
    }
