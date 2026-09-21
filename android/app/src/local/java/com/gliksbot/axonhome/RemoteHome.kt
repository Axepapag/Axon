package com.gliksbot.axonhome

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.delay

@Composable
fun RemoteHome() {
    val context = LocalContext.current
    val store = remember { RemoteConnectionStore(context.applicationContext) }
    var enrollment by remember { mutableStateOf(store.load()) }
    var endpoint by remember { mutableStateOf(enrollment?.baseUrl.orEmpty()) }
    var token by remember { mutableStateOf("") }
    var status by remember { mutableStateOf("Not connected") }
    var summary by remember { mutableStateOf<RemoteControlClient.RuntimeSummary?>(null) }
    var error by remember { mutableStateOf<String?>(null) }
    var showRaw by remember { mutableStateOf(false) }

    LaunchedEffect(enrollment) {
        val active = enrollment ?: return@LaunchedEffect
        val client = RemoteControlClient(active)
        while (true) {
            try {
                summary = client.runtimeSummary()
                status = "Connected"
                error = null
            } catch (t: Throwable) {
                status = "Reconnecting"
                error = t.message ?: t::class.java.simpleName
            }
            delay(5_000)
        }
    }

    MaterialTheme(colorScheme = darkColorScheme()) {
        Surface(Modifier.fillMaxSize()) {
            Column(
                Modifier.safeDrawingPadding().verticalScroll(rememberScrollState()).padding(18.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Text("Axon Home", style = MaterialTheme.typography.headlineLarge)
                Text("CLOUD VM CONTROL CENTER", color = MaterialTheme.colorScheme.primary)
                Text(status)

                if (enrollment == null) {
                    Text("Enroll this phone once. Axon Home stores the control credential with Android Keystore and reconnects automatically after that.")
                    OutlinedTextField(
                        value = endpoint,
                        onValueChange = { endpoint = it },
                        label = { Text("VM control URL (https://…)") },
                        modifier = Modifier.fillMaxWidth(),
                        singleLine = true,
                    )
                    OutlinedTextField(
                        value = token,
                        onValueChange = { token = it },
                        label = { Text("One-time control token") },
                        visualTransformation = PasswordVisualTransformation(),
                        modifier = Modifier.fillMaxWidth(),
                        singleLine = true,
                    )
                    Button(onClick = {
                        try {
                            store.save(endpoint, token)
                            enrollment = store.load()
                            token = ""
                            error = null
                        } catch (t: Throwable) {
                            error = t.message
                        }
                    }) { Text("Enroll VM") }
                } else {
                    Text(enrollment!!.baseUrl, style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace)
                    summary?.let { s ->
                        HorizontalDivider()
                        Text("Heart / body", style = MaterialTheme.typography.titleLarge)
                        Metric("Status", s.controlStatus)
                        Metric("Heartbeat", s.heartbeatSequence?.toString() ?: "—")
                        Metric("Reasoning tick", s.tickSequence?.toString() ?: "—")
                        Metric("Canonical field", compactId(s.fieldId), mono = true)
                        Metric("Attended D64 view", compactId(s.viewId), mono = true)

                        HorizontalDivider()
                        Text("Cortex / ingress", style = MaterialTheme.typography.titleLarge)
                        Metric("Dormant index", compactId(s.dormantIndexId), mono = true)
                        Metric("Ingress quarantined", s.ingressQuarantineCount?.toString() ?: "—")
                        Metric("Ingress rejected", s.ingressRejectionCount?.toString() ?: "—")

                        HorizontalDivider()
                        Text("Trainer", style = MaterialTheme.typography.titleLarge)
                        Metric("Lifecycle", s.trainerStatus)
                        Metric("Candidate", compactId(s.candidateGeneration), mono = true)
                        Metric("Accepted step", s.trainingStep?.toString() ?: "—")
                        Metric("Loss", s.loss?.let { "%.6f".format(it) } ?: "—", mono = true)
                        Metric("Progress", s.trainingProgressStatus ?: "—")

                        TextButton(onClick = { showRaw = !showRaw }) {
                            Text(if (showRaw) "Hide raw runtime JSON" else "Show raw runtime JSON")
                        }
                        if (showRaw) {
                            Text(s.raw, style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace)
                        }
                    }
                    OutlinedButton(onClick = {
                        store.clear()
                        enrollment = null
                        summary = null
                        status = "Not connected"
                    }) { Text("Forget VM enrollment") }
                }

                error?.let { Text("Connection: $it", color = MaterialTheme.colorScheme.error) }
                Text(
                    "Server authoritative: Heart, canonical State, D16/D64 Rails, Cortex, Souls and Trainer stay on the cloud VM. Axon Home observes and controls through governed APIs rather than SSH.",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }
    }
}

private fun compactId(value: String?): String = when {
    value.isNullOrBlank() -> "—"
    value.length <= 20 -> value
    else -> value.take(10) + "…" + value.takeLast(8)
}

@Composable
private fun Metric(label: String, value: String, mono: Boolean = false) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label)
        Text(
            value,
            style = MaterialTheme.typography.bodyMedium,
            fontFamily = if (mono) FontFamily.Monospace else FontFamily.Default,
        )
    }
}
