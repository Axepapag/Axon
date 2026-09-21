package com.gliksbot.axonhome.ui.field

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.gliksbot.axonhome.controlplane.LogicalRegion
import com.gliksbot.axonhome.controlplane.MaskKind
import com.gliksbot.axonhome.controlplane.MaskPolicy
import com.gliksbot.axonhome.ui.components.Chip
import com.gliksbot.axonhome.ui.components.SectionCard
import com.gliksbot.axonhome.ui.state.maskPolicyLabel
import com.gliksbot.axonhome.ui.theme.SimAccent
import kotlin.math.roundToInt

/**
 * Attention / Region mask controls (arch §10.F). Masks are a DERIVED VIEW —
 * production changes go through the Control Plane two-step confirm
 * (PUT /v1/masks proposal → confirm). Wired to the sim mask controller via
 * SimControlPlaneClient; nothing is applied on first touch.
 */
@Composable
fun MaskScreen(viewModel: FieldViewModel, onBack: () -> Unit) {
    val mask by viewModel.maskState.collectAsStateWithLifecycle()
    val current = mask.current

    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp)) {
        Text("Attention / Region masks", style = MaterialTheme.typography.titleLarge)
        Text(
            "Masks are a derived view; they never mutate the canonical field. " +
                "Production change requires the Control Plane two-step confirm — " +
                "here it is wired to the simulator's mask controller.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        if (current != null) {
            Text(
                "revision ${current.revision} · state ${current.stateId.take(16)}… · view ${current.viewId.take(16)}…",
                style = MaterialTheme.typography.labelSmall,
                fontFamily = FontFamily.Monospace,
                modifier = Modifier.padding(vertical = 4.dp),
            )
        }

        mask.draft.forEach { (region, policy) ->
            val currentPolicy = current?.policies?.get(region)
            val dirty = currentPolicy != policy
            SectionCard(region.name.lowercase(), Modifier.padding(top = 8.dp)) {
                Row(verticalAlignment = androidx.compose.ui.Alignment.CenterVertically) {
                    Chip("current: ${maskPolicyLabel(currentPolicy ?: policy)}", SimAccent)
                    if (dirty) Chip("proposed: ${maskPolicyLabel(policy)}", MaterialTheme.colorScheme.primary)
                }
                Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                    listOf("all", "none", "tail").forEach { kind ->
                        OutlinedButton(onClick = {
                            when (kind) {
                                "all" -> viewModel.draftMask(region, MaskPolicy(MaskKind.ALL))
                                "none" -> viewModel.draftMask(region, MaskPolicy(MaskKind.NONE))
                                else -> viewModel.draftMask(
                                    region,
                                    MaskPolicy(MaskKind.TAIL_PERCENT, tailPercent = 50),
                                )
                            }
                        }) { Text(kind) }
                    }
                }
                if (policy.kind == MaskKind.TAIL_PERCENT) {
                    val value = (policy.tailPercent ?: 0).toFloat()
                    Slider(
                        value = value,
                        onValueChange = {
                            viewModel.draftMask(
                                region,
                                MaskPolicy(MaskKind.TAIL_PERCENT, tailPercent = it.roundToInt().coerceIn(0, 100)),
                            )
                        },
                        valueRange = 0f..100f,
                    )
                    Text("newest ${value.roundToInt()}% of region text", style = MaterialTheme.typography.labelSmall)
                }
            }
        }

        Row(Modifier.padding(top = 12.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = { viewModel.proposeMasks() }, enabled = !mask.busy) {
                Text("Review proposed changes")
            }
            OutlinedButton(onClick = { viewModel.revertMaskDraft() }) { Text("Revert to current") }
        }
        if (mask.applied) {
            Text("Applied (audited).", color = MaterialTheme.colorScheme.secondary, modifier = Modifier.padding(top = 8.dp))
        }
        mask.error?.let {
            Text(it, color = MaterialTheme.colorScheme.error, modifier = Modifier.padding(top = 8.dp))
        }
    }

    if (mask.confirmToken != null) {
        AlertDialog(
            onDismissRequest = { viewModel.revertMaskDraft() },
            title = { Text("Confirm mask change") },
            text = {
                Column {
                    Text("Proposed changes (production path: Control Plane two-step confirm):")
                    mask.proposalChanges.forEach { Text("• $it", style = MaterialTheme.typography.bodySmall) }
                    if (mask.proposalChanges.isEmpty()) Text("• (no effective changes)")
                }
            },
            confirmButton = {
                Button(
                    onClick = { viewModel.confirmMasks() },
                    colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                ) { Text("Apply (confirm)") }
            },
            dismissButton = {
                OutlinedButton(onClick = { viewModel.revertMaskDraft() }) { Text("Cancel") }
            },
        )
    }
}
