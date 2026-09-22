package com.gliksbot.axonhome.ui.trainer

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
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.gliksbot.axonhome.controlplane.TrainerCommandKind
import com.gliksbot.axonhome.controlplane.StopScope
import com.gliksbot.axonhome.ui.components.Chip
import com.gliksbot.axonhome.ui.components.FreshnessChip
import com.gliksbot.axonhome.ui.components.InfoRow
import com.gliksbot.axonhome.ui.components.SectionCard
import com.gliksbot.axonhome.ui.components.Sparkline
import com.gliksbot.axonhome.ui.theme.AxonError

@Composable
fun TrainerScreen(viewModel: TrainerViewModel) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val stopSummary by viewModel.stopSummary.collectAsStateWithLifecycle()
    var stopDialogOpen by remember { mutableStateOf(false) }

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Row(verticalAlignment = androidx.compose.ui.Alignment.CenterVertically) {
            FreshnessChip(state.freshness, simulation = true)
            Chip(
                state.lifecycle,
                if (state.lifecycle == "RUNNING") Color(0xFF2E7D32) else Color(0xFF9E9E9E),
            )
        }

        SectionCard("Lineage") {
            InfoRow("Module", state.moduleId.ifEmpty { "—" }, mono = true)
            InfoRow("Base generation", state.baseGeneration.ifEmpty { "—" }, mono = true)
            InfoRow("Candidate generation", state.candidateGeneration.ifEmpty { "—" }, mono = true)
            InfoRow("Accepted global step", state.globalStep.toString(), mono = true)
            InfoRow("Optimizer", state.optimizer.ifEmpty { "—" })
            InfoRow("Learning rate", if (state.learningRate > 0) state.learningRate.toString() else "—", mono = true)
        }

        SectionCard("Tranche") {
            if (state.trancheFinal > state.trancheBase) {
                InfoRow("Window", "[${state.trancheBase}, ${state.trancheFinal})", mono = true)
                InfoRow("Progress", "${state.globalStep - state.trancheBase} / ${state.trancheFinal - state.trancheBase} steps")
                InfoRow("Purpose", state.tranchePurpose)
            } else {
                Text("No active tranche.")
            }
        }

        SectionCard("Loss (accepted steps)") {
            if (state.losses.size >= 2) {
                Sparkline(state.losses)
                Text(
                    "latest ${"%.4f".format(state.losses.last())} · n=${state.losses.size}",
                    style = MaterialTheme.typography.labelSmall,
                    fontFamily = FontFamily.Monospace,
                )
            } else {
                Text("No accepted steps yet — start training.")
            }
        }

        SectionCard("Gradient L2 norm") {
            if (state.gradNorms.size >= 2) {
                Sparkline(state.gradNorms, lineColor = MaterialTheme.colorScheme.tertiary)
            } else {
                Text("No gradient telemetry yet.")
            }
        }

        SectionCard("Checkpoints (rolling-3 retention)") {
            state.retainedCheckpoints.forEach { Text(it, style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace) }
            if (state.retainedCheckpoints.isEmpty()) Text("None yet.")
            InfoRow("Heldout evaluations", state.heldoutEvals.toString())
        }

        SectionCard("Workers (sim)") {
            state.workers.forEach { (id, connected) ->
                Row {
                    Text(id, style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace, modifier = Modifier.weight(1f))
                    Chip(if (connected) "connected" else "lost", if (connected) Color(0xFF2E7D32) else AxonError)
                }
            }
        }

        SectionCard("Controls") {
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                OutlinedButton(onClick = { viewModel.command(TrainerCommandKind.PREFLIGHT) }) { Text("Preflight") }
                Button(onClick = { viewModel.command(TrainerCommandKind.START) }) { Text("Start / resume") }
                OutlinedButton(onClick = { viewModel.command(TrainerCommandKind.PAUSE) }) { Text("Pause at boundary") }
            }
            state.commandFeedback?.let {
                Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.secondary)
            }
        }

        // Global red STOP (arch §10.J / §13).
        Button(
            onClick = { stopDialogOpen = true },
            colors = ButtonDefaults.buttonColors(containerColor = AxonError),
            modifier = Modifier.fillMaxWidth(),
        ) { Text("STOP") }
    }

    if (stopDialogOpen) {
        AlertDialog(
            onDismissRequest = { stopDialogOpen = false },
            title = { Text("Stop semantics") },
            text = {
                Text(
                    "SAFE stops at the next atomic boundary and preserves recoverability " +
                        "(checkpoints, souls, pointers). EMERGENCY attempts immediate halt " +
                        "and reports per-component acknowledgements — unreachable components " +
                        "are shown as unreachable, never as stopped.",
                )
            },
            confirmButton = {
                Button(onClick = {
                    stopDialogOpen = false
                    viewModel.stop(StopScope.STOP_TRAINING)
                }) { Text("Safe stop (boundary)") }
            },
            dismissButton = {
                Button(
                    onClick = {
                        stopDialogOpen = false
                        viewModel.stop(StopScope.EMERGENCY)
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = AxonError),
                ) { Text("EMERGENCY stop all") }
            },
        )
    }

    stopSummary?.let { summary ->
        AlertDialog(
            onDismissRequest = { viewModel.dismissStopSummary() },
            title = { Text("Stop result — ${summary.scopeLabel}") },
            text = {
                Column {
                    Text(
                        summary.headline,
                        style = MaterialTheme.typography.titleSmall,
                        color = if (summary.allStopped && summary.claimVerified) Color(0xFF2E7D32) else AxonError,
                    )
                    if (!summary.claimVerified) {
                        Text(
                            "The response's all_stopped claim disagreed with its own ack map; " +
                                "the derived value is shown.",
                            style = MaterialTheme.typography.bodySmall,
                            color = AxonError,
                        )
                    }
                    summary.rows.forEach { row ->
                        Row(Modifier.padding(top = 6.dp)) {
                            Column(Modifier.weight(1f)) {
                                Text(row.componentId, style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace)
                                row.detail?.let { Text(it, style = MaterialTheme.typography.labelSmall) }
                            }
                            Chip(
                                row.statusLabel,
                                if (row.blocksAllStopped) AxonError else Color(0xFF2E7D32),
                            )
                        }
                    }
                }
            },
            confirmButton = {
                Button(onClick = { viewModel.dismissStopSummary() }) { Text("Close") }
            },
        )
    }
}
