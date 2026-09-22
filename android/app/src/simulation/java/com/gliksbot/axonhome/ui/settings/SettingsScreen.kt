package com.gliksbot.axonhome.ui.settings

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.gliksbot.axonhome.BuildConfig
import com.gliksbot.axonhome.data.ThemeMode
import com.gliksbot.axonhome.ui.components.InfoRow
import com.gliksbot.axonhome.ui.components.SectionCard
import kotlin.math.roundToLong

@Composable
fun SettingsScreen(viewModel: SettingsViewModel, onOpenAudit: () -> Unit) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val context = LocalContext.current

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        SectionCard("Appearance") {
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                ThemeMode.entries.forEach { mode ->
                    FilterChip(
                        selected = state.themeMode == mode,
                        onClick = { viewModel.setTheme(mode) },
                        label = { Text(mode.name.lowercase()) },
                    )
                }
            }
        }

        SectionCard("Refresh / simulation cadence") {
            Text("Tick interval: ${state.tickIntervalMs} ms", style = MaterialTheme.typography.bodySmall)
            Slider(
                value = state.tickIntervalMs.toFloat(),
                onValueChange = { viewModel.setTickInterval(it.roundToLong()) },
                valueRange = 500f..10_000f,
            )
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Auto-advance ticks", Modifier.weight(1f))
                Switch(checked = state.autoTick, onCheckedChange = { viewModel.setAutoTick(it) })
            }
        }

        SectionCard("Security") {
            Text("Vault auto-lock: ${state.autoLockSeconds}s", style = MaterialTheme.typography.bodySmall)
            Slider(
                value = state.autoLockSeconds.toFloat(),
                onValueChange = { viewModel.setAutoLock(it.roundToLong().toInt()) },
                valueRange = 0f..600f,
            )
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Biometric / device-auth unlock", Modifier.weight(1f))
                Switch(checked = state.biometricEnabled, onCheckedChange = { viewModel.setBiometric(it) })
            }
            Text(
                "TLS certificate errors are never silently accepted; there is no " +
                    "accept-all-certs toggle (arch §8).",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        SectionCard("Endpoints") {
            Text(
                "Endpoint profiles are managed on the Compute screen. The built-in " +
                    "simulator profile is always available.",
                style = MaterialTheme.typography.bodySmall,
            )
        }

        SectionCard("Data") {
            Button(onClick = { viewModel.exportConfig(context) }) { Text("Export non-secret config (JSON)") }
            OutlinedButton(onClick = onOpenAudit, Modifier.padding(top = 6.dp)) { Text("Operational audit log") }
        }

        SectionCard("About") {
            InfoRow("App", "Axon Home")
            InfoRow("Version", BuildConfig.VERSION_NAME)
            InfoRow("Git commit", BuildConfig.GIT_COMMIT, mono = true)
            InfoRow("Sim seed", BuildConfig.SIM_SEED, mono = true)
            Text(
                "SIMULATION build: all organism data is synthetic and deterministic.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.tertiary,
                fontFamily = FontFamily.Monospace,
            )
        }
    }
}
