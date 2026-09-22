package com.gliksbot.axonhome.ui.storage

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
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.gliksbot.axonhome.ui.components.Chip
import com.gliksbot.axonhome.ui.components.FreshnessChip
import com.gliksbot.axonhome.ui.components.InfoRow
import com.gliksbot.axonhome.ui.components.SectionCard
import com.gliksbot.axonhome.ui.theme.AxonError

@Composable
fun StorageScreen(viewModel: StorageViewModel) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val verify by viewModel.verifySummary.collectAsStateWithLifecycle()

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        FreshnessChip(state.freshness, simulation = true)

        SectionCard("Storage profiles") {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("Google Drive", style = MaterialTheme.typography.bodyMedium)
                    Text(
                        "Architecture prepared — not implemented (Phase 3, arch §18)",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.tertiary,
                    )
                }
                Chip("planned", MaterialTheme.colorScheme.tertiary)
            }
            HorizontalDivider(Modifier.padding(vertical = 6.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("SFTP", style = MaterialTheme.typography.bodyMedium)
                    Text("Adapter contract designed (arch §7); not implemented — Phase 4", style = MaterialTheme.typography.bodySmall)
                }
                Chip("planned", MaterialTheme.colorScheme.tertiary)
            }
            HorizontalDivider(Modifier.padding(vertical = 6.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("Local (sim artifact catalog)", style = MaterialTheme.typography.bodyMedium)
                    Text("Metadata mirror only — hashes and records, no bulk payloads", style = MaterialTheme.typography.bodySmall)
                }
                Chip("active (sim)", Color(0xFF2E7D32))
            }
        }

        SectionCard("Artifact catalog (sim)") {
            if (state.artifacts.isEmpty()) {
                Text("No artifacts yet — checkpoints appear once training accepts steps.")
            }
            state.artifacts.forEach { artifact ->
                Column(Modifier.padding(vertical = 4.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(artifact.kind.name.lowercase(), style = MaterialTheme.typography.labelLarge, modifier = Modifier.weight(1f))
                        Chip(artifact.provider, MaterialTheme.colorScheme.primary)
                    }
                    Text(artifact.path, style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace)
                    Text(
                        "sha256 ${artifact.sha256.take(20)}… · ${artifact.bytes} B" +
                            (artifact.verifiedAt?.let { " · verified $it" } ?: " · not verified"),
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }

        SectionCard("Recovery capsules") {
            state.capsules.forEach { capsule ->
                Row(Modifier.padding(vertical = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text(capsule.capsuleId.take(28) + "…", fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodySmall)
                        Text(
                            "${capsule.kind.name.lowercase()} · ${capsule.verificationStatus.name.lowercase()}",
                            style = MaterialTheme.typography.labelSmall,
                        )
                    }
                    OutlinedButton(
                        onClick = { viewModel.verifyCapsule(capsule.capsuleId) },
                        enabled = !state.verifying,
                    ) { Text("Verify") }
                }
            }
            if (state.capsules.isEmpty()) Text("No capsules exported yet.")
            Button(
                onClick = { viewModel.exportCapsule() },
                modifier = Modifier.padding(top = 8.dp),
            ) { Text("Export recovery capsule (sim)") }
        }

        state.feedback?.let {
            Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.secondary)
        }
    }

    verify?.let { summary ->
        AlertDialog(
            onDismissRequest = { viewModel.dismissVerify() },
            title = { Text("Capsule verification") },
            text = {
                Column(Modifier.verticalScroll(rememberScrollState())) {
                    Text(
                        summary.headline,
                        style = MaterialTheme.typography.titleSmall,
                        color = if (summary.ok) Color(0xFF2E7D32) else AxonError,
                    )
                    summary.rows.forEach { row ->
                        Row(Modifier.padding(top = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                            Text(row.name, Modifier.weight(1f), style = MaterialTheme.typography.bodySmall)
                            Chip(if (row.ok) "pass" else "FAIL", if (row.ok) Color(0xFF2E7D32) else AxonError)
                        }
                        row.detail?.let { Text(it, style = MaterialTheme.typography.labelSmall) }
                    }
                    summary.mismatches.forEach {
                        Text("✗ $it", color = AxonError, style = MaterialTheme.typography.bodySmall)
                    }
                }
            },
            confirmButton = { Button(onClick = { viewModel.dismissVerify() }) { Text("Close") } },
        )
    }
}
