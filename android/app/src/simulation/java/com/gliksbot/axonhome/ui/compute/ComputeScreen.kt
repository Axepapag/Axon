package com.gliksbot.axonhome.ui.compute

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
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
import com.gliksbot.axonhome.controlplane.WorkerStatus
import com.gliksbot.axonhome.data.EndpointProfile
import com.gliksbot.axonhome.ui.components.Chip
import com.gliksbot.axonhome.ui.components.CostConfirmDialog
import com.gliksbot.axonhome.ui.components.FreshnessChip
import com.gliksbot.axonhome.ui.components.InfoRow
import com.gliksbot.axonhome.ui.components.SectionCard

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun ComputeScreen(viewModel: ComputeViewModel) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    var newEndpointName by remember { mutableStateOf("") }
    var newEndpointUrl by remember { mutableStateOf("") }

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        FreshnessChip(state.freshness, simulation = true)

        SectionCard("Workers (sim)") {
            if (state.workers.isEmpty()) Text("No workers.")
            state.workers.forEach { worker ->
                Column(Modifier.padding(vertical = 6.dp)) {
                    Row(verticalAlignment = androidx.compose.ui.Alignment.CenterVertically) {
                        Text(worker.workerId, fontFamily = FontFamily.Monospace, modifier = Modifier.weight(1f))
                        Chip(
                            worker.status.name.lowercase(),
                            when (worker.status) {
                                WorkerStatus.CONNECTED, WorkerStatus.BUSY -> Color(0xFF2E7D32)
                                WorkerStatus.LOST -> MaterialTheme.colorScheme.error
                                else -> Color(0xFF9E9E9E)
                            },
                        )
                    }
                    InfoRow("Provider", worker.provider)
                    InfoRow("Endpoint", worker.endpoint, mono = true)
                    InfoRow("Machine", worker.accelerator ?: "cpu-only")
                    InfoRow("Active job", worker.activeJobId?.take(24)?.plus("…") ?: "—", mono = true)
                    InfoRow("Cost", "unknown (sim provider does not advertise pricing)")
                    FlowRow {
                        worker.capabilities.advertised().sorted().forEach { cap ->
                            Chip(cap, MaterialTheme.colorScheme.primary)
                        }
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        if (worker.capabilities.supportsStop && worker.status == WorkerStatus.CONNECTED) {
                            OutlinedButton(onClick = { viewModel.stopWorker(worker.workerId) }) { Text("Stop") }
                        }
                        OutlinedButton(onClick = {
                            viewModel.requestBillableAction("provision new worker via ${worker.provider}")
                        }) { Text("Provision…") }
                    }
                }
            }
        }

        SectionCard("Endpoint profiles (local)") {
            state.endpoints.forEach { profile ->
                Row(Modifier.padding(vertical = 3.dp), verticalAlignment = androidx.compose.ui.Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text(profile.name, style = MaterialTheme.typography.bodyMedium)
                        Text(profile.baseUrl, style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace)
                    }
                    if (profile.isSimulation) Chip("active", MaterialTheme.colorScheme.secondary)
                    else OutlinedButton(onClick = { viewModel.deleteEndpoint(profile.id) }) { Text("Delete") }
                }
            }
            Row(Modifier.padding(top = 8.dp), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                OutlinedTextField(
                    value = newEndpointName,
                    onValueChange = { newEndpointName = it },
                    label = { Text("Name") },
                    modifier = Modifier.weight(1f),
                )
                OutlinedTextField(
                    value = newEndpointUrl,
                    onValueChange = { newEndpointUrl = it },
                    label = { Text("Base URL") },
                    modifier = Modifier.weight(1.4f),
                )
            }
            Button(
                onClick = {
                    if (newEndpointName.isNotBlank() && newEndpointUrl.isNotBlank()) {
                        viewModel.saveEndpoint(
                            EndpointProfile(
                                id = "ep-${System.currentTimeMillis()}",
                                name = newEndpointName.trim(),
                                baseUrl = newEndpointUrl.trim(),
                            )
                        )
                        newEndpointName = ""
                        newEndpointUrl = ""
                    }
                },
                modifier = Modifier.padding(top = 6.dp),
                enabled = newEndpointName.isNotBlank() && newEndpointUrl.isNotBlank(),
            ) { Text("Save profile") }
            Text(
                "Profiles are stored locally (DataStore/filesDir JSON). Tokens are vault " +
                    "references only — never raw secrets.",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        state.feedback?.let {
            Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.secondary)
        }
    }

    state.pendingBillableAction?.let { action ->
        CostConfirmDialog(
            title = "Confirm billable action",
            provider = "sim provider",
            costHint = "unknown — provider does not advertise pricing in the simulator",
            onConfirm = { viewModel.confirmBillableAction() },
            onDismiss = { viewModel.cancelBillableAction() },
        )
    }
}
