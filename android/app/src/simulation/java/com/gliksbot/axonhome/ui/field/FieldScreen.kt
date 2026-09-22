package com.gliksbot.axonhome.ui.field

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.HorizontalDivider
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
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.gliksbot.axonhome.ui.components.Chip
import com.gliksbot.axonhome.ui.components.FreshnessChip
import com.gliksbot.axonhome.ui.components.InfoRow
import com.gliksbot.axonhome.ui.components.SectionCard
import com.gliksbot.axonhome.ui.components.TwoPane
import com.gliksbot.axonhome.ui.theme.SimAccent

@Composable
fun FieldScreen(viewModel: FieldViewModel, expanded: Boolean, onOpenMasks: () -> Unit) {
    val state by viewModel.state.collectAsStateWithLifecycle()

    val listPane: @Composable () -> Unit = {
        RegionListPane(state, viewModel, onOpenMasks)
    }
    val detailPane: @Composable () -> Unit = {
        RegionDetailPane(state)
    }

    if (expanded) {
        TwoPane(expanded = true, first = listPane, second = detailPane)
    } else {
        Column(Modifier.fillMaxSize()) {
            RegionListPane(state, viewModel, onOpenMasks, modifier = Modifier.weight(1f))
        }
    }
}

@Composable
private fun RegionListPane(
    state: FieldState,
    viewModel: FieldViewModel,
    onOpenMasks: () -> Unit,
    modifier: Modifier = Modifier,
) {
    var showRaw by remember { mutableStateOf(false) }
    var selected by remember { mutableStateOf<Int?>(null) }

    LazyColumn(modifier.fillMaxSize().padding(horizontal = 12.dp)) {
        item {
            Row(
                Modifier.fillMaxWidth().padding(vertical = 4.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = androidx.compose.ui.Alignment.CenterVertically,
            ) {
                FreshnessChip(state.freshness, simulation = true)
                Row {
                    OutlinedButton(
                        onClick = { viewModel.viewTickOffset(state.viewingTickOffset + 1) },
                        enabled = state.viewingTickOffset < state.retainedTicks - 1,
                    ) { Text("◀ tick") }
                    OutlinedButton(
                        onClick = { viewModel.viewTickOffset(state.viewingTickOffset - 1) },
                        enabled = state.viewingTickOffset > 0,
                        modifier = Modifier.padding(start = 4.dp),
                    ) { Text("tick ▶") }
                }
            }
            Text(
                "Tick ${state.tickId} · field ${state.fieldId.take(16)}… " +
                    (if (state.viewingTickOffset > 0) "(retained history — read-only)" else "(head)"),
                style = MaterialTheme.typography.labelMedium,
                fontFamily = FontFamily.Monospace,
            )
            Text(
                "canonical sha256 ${state.canonicalHash.take(24)}… · ${state.retainedTicks} ticks retained",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                fontFamily = FontFamily.Monospace,
            )
        }

        item {
            SectionCard("13 canonical regions", Modifier.padding(top = 8.dp)) {
                state.regions.forEachIndexed { index, region ->
                    Row(
                        Modifier
                            .fillMaxWidth()
                            .clickable { selected = index; viewModel.selectRegion(index) }
                            .padding(vertical = 4.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text(
                                "${index + 1}. ${region.wireName}",
                                style = MaterialTheme.typography.bodyMedium,
                                fontFamily = FontFamily.Monospace,
                            )
                            Text(
                                "${region.charCount} chars · ${region.spanCount} spans",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                        Chip("mask: ${region.maskLabel}", SimAccent)
                    }
                    if (index < state.regions.size - 1) HorizontalDivider()
                }
            }
        }

        item {
            SectionCard("Tick pipeline (latest retained tick)", Modifier.padding(top = 8.dp)) {
                val pipeline = state.pipeline
                if (pipeline == null) {
                    Text("No ticks yet.")
                } else {
                    Text(
                        "F_t ${pipeline.baseFieldId.take(12)}…  →  tick ${pipeline.tickNumber}",
                        style = MaterialTheme.typography.labelMedium,
                        fontFamily = FontFamily.Monospace,
                    )
                    Text("FIRST board (${pipeline.firstBoard.size} proposals):", style = MaterialTheme.typography.labelLarge)
                    pipeline.firstBoard.forEach { (core, text) ->
                        Text("• $core: $text", style = MaterialTheme.typography.bodySmall)
                    }
                    Text("REFINED board (${pipeline.refinedBoard.size}):", style = MaterialTheme.typography.labelLarge)
                    pipeline.refinedBoard.forEach { (core, text) ->
                        Text("• $core: $text", style = MaterialTheme.typography.bodySmall)
                    }
                    Text("Consolidator: ${pipeline.consolidator}", style = MaterialTheme.typography.labelLarge)
                    if (pipeline.rejected) {
                        Text(
                            "FINAL REJECTED by Heart validation: ${pipeline.rejectionReason}",
                            color = MaterialTheme.colorScheme.error,
                            style = MaterialTheme.typography.bodySmall,
                        )
                    } else {
                        Text(
                            "FINAL: ${pipeline.verdictText?.take(200) ?: "—"}",
                            style = MaterialTheme.typography.bodySmall,
                        )
                        Text(
                            "FieldDelta → ${pipeline.deltaIds.joinToString { it.take(12) + "…" }} → F_{t+1} ${pipeline.resultFieldId?.take(12)}…",
                            style = MaterialTheme.typography.bodySmall,
                            fontFamily = FontFamily.Monospace,
                        )
                    }
                }
            }
        }

        item {
            SectionCard("User input → user_input region", Modifier.padding(top = 8.dp)) {
                OutlinedTextField(
                    value = state.inputDraft,
                    onValueChange = { viewModel.setInputDraft(it) },
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("Say something to the organism") },
                )
                Row(Modifier.padding(top = 6.dp)) {
                    Button(onClick = { viewModel.submitUserInput() }, enabled = state.inputDraft.isNotBlank()) {
                        Text("Submit to field")
                    }
                    OutlinedButton(onClick = onOpenMasks, Modifier.padding(start = 8.dp)) {
                        Text("Mask controls")
                    }
                }
                state.inputFeedback?.let {
                    Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.secondary)
                }
            }
        }

        item {
            SectionCard("Raw snapshot JSON", Modifier.padding(top = 8.dp), onClick = { showRaw = !showRaw }) {
                Text(if (showRaw) "Tap to collapse" else "Tap to expand", style = MaterialTheme.typography.labelSmall)
                if (showRaw) {
                    Text(
                        state.rawJson.take(8000),
                        style = MaterialTheme.typography.bodySmall,
                        fontFamily = FontFamily.Monospace,
                    )
                }
            }
        }
    }

    selected?.let { index ->
        val region = state.regions.getOrNull(index)
        if (region != null) {
            androidx.compose.material3.AlertDialog(
                onDismissRequest = { selected = null },
                title = { Text(region.wireName, fontFamily = FontFamily.Monospace) },
                text = {
                    Column(Modifier.verticalScroll(rememberScrollState())) {
                        InfoRow("Chars", region.charCount.toString(), mono = true)
                        InfoRow("Mask", region.maskLabel)
                        InfoRow("Spans", region.spanCount.toString())
                        HorizontalDivider(Modifier.padding(vertical = 6.dp))
                        if (region.spans.isEmpty()) {
                            Text("(empty region)")
                        }
                        region.spans.forEach { span ->
                            Text("span ${span.spanId.take(12)}… · ${span.kind}", style = MaterialTheme.typography.labelMedium)
                            Text("source: ${span.source.ifEmpty { "—" }}", style = MaterialTheme.typography.bodySmall)
                            Text("provenance: ${span.provenance.ifEmpty { "—" }}", style = MaterialTheme.typography.bodySmall)
                            Text(span.textPreview, style = MaterialTheme.typography.bodySmall)
                            HorizontalDivider(Modifier.padding(vertical = 6.dp))
                        }
                    }
                },
                confirmButton = {
                    OutlinedButton(onClick = { selected = null }) { Text("Close") }
                },
            )
        }
    }
}

@Composable
private fun RegionDetailPane(state: FieldState) {
    val region = state.regions.getOrNull(state.selectedRegion)
    Column(Modifier.fillMaxSize().padding(12.dp).verticalScroll(rememberScrollState())) {
        if (region == null) {
            Text("Select a region")
        } else {
            Text(region.wireName, style = MaterialTheme.typography.titleMedium, fontFamily = FontFamily.Monospace)
            InfoRow("Chars", region.charCount.toString(), mono = true)
            InfoRow("Mask", region.maskLabel)
            HorizontalDivider(Modifier.padding(vertical = 6.dp))
            region.spans.forEach { span ->
                SectionCard("span ${span.spanId.take(12)}…") {
                    InfoRow("Kind", span.kind)
                    InfoRow("Source", span.source.ifEmpty { "—" })
                    InfoRow("Provenance", span.provenance.ifEmpty { "—" })
                    Text(span.textPreview, style = MaterialTheme.typography.bodySmall)
                }
            }
            if (region.spans.isEmpty()) Text("(empty region)")
        }
    }
}
