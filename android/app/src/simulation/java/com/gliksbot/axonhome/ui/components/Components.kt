package com.gliksbot.axonhome.ui.components

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.gliksbot.axonhome.controlplane.Freshness
import com.gliksbot.axonhome.ui.state.freshnessLabel
import com.gliksbot.axonhome.ui.theme.SimAccent

/**
 * Persistent, unmistakable SIMULATION banner (arch §11). Rendered at the top
 * of every screen while the app is backed by the simulator.
 */
@Composable
fun SimulationBanner(modifier: Modifier = Modifier) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .background(SimAccent)
            .padding(horizontal = 12.dp, vertical = 6.dp),
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            "SIMULATION — deterministic synthetic organism, NOT live Axon",
            color = Color(0xFF1A1200),
            style = MaterialTheme.typography.labelLarge,
            fontWeight = FontWeight.Bold,
        )
    }
}

/** Freshness chip (arch §12): LIVE / CACHED / STALE / DISCONNECTED (+ SIMULATION prefix). */
@Composable
fun FreshnessChip(
    freshness: Freshness,
    asOf: String? = null,
    simulation: Boolean = true,
    modifier: Modifier = Modifier,
) {
    val color = when (freshness) {
        Freshness.LIVE -> Color(0xFF2E7D32)
        Freshness.CACHED -> Color(0xFF1565C0)
        Freshness.STALE -> Color(0xFFEF6C00)
        Freshness.DISCONNECTED -> Color(0xFF9E9E9E)
    }
    Surface(
        color = color.copy(alpha = 0.25f),
        shape = MaterialTheme.shapes.small,
        modifier = modifier,
    ) {
        Text(
            text = freshnessLabel(freshness, simulation) + (asOf?.let { " · $it" } ?: ""),
            color = color,
            style = MaterialTheme.typography.labelSmall,
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
        )
    }
}

@Composable
fun SectionCard(
    title: String,
    modifier: Modifier = Modifier,
    onClick: (() -> Unit)? = null,
    content: @Composable () -> Unit,
) {
    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        onClick = { onClick?.invoke() },
        enabled = onClick != null,
    ) {
        Column(Modifier.padding(12.dp)) {
            Text(title, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
            Spacer(Modifier.height(6.dp))
            content()
        }
    }
}

@Composable
fun InfoRow(label: String, value: String, mono: Boolean = false) {
    Row(Modifier.fillMaxWidth().padding(vertical = 1.dp)) {
        Text(
            label,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.weight(0.42f),
        )
        Text(
            value,
            style = MaterialTheme.typography.bodySmall,
            fontFamily = if (mono) FontFamily.Monospace else null,
            modifier = Modifier.weight(0.58f),
        )
    }
}

/** Honest stub for controls not wired to a governed seam (arch §10). */
@Composable
fun NotImplementedButton(label: String) {
    TextButton(onClick = { }, enabled = false) {
        Text("$label — Not implemented")
    }
}

/** Full-screen honest stub for screens outside this slice (arch §18 phases). */
@Composable
fun StubScreen(title: String, phase: String, detail: String) {
    Column(
        modifier = Modifier.fillMaxSize().padding(24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(title, style = MaterialTheme.typography.headlineSmall)
        Spacer(Modifier.height(8.dp))
        Text(
            "Not implemented yet — $phase",
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.tertiary,
        )
        Spacer(Modifier.height(12.dp))
        Text(
            detail,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

/**
 * Cost-confirmation dialog (arch §8): provider + expected cost must be shown
 * before any billable action. The action itself may be an honest stub.
 */
@Composable
fun CostConfirmDialog(
    title: String,
    provider: String,
    costHint: String,
    onConfirm: () -> Unit,
    onDismiss: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(title) },
        text = {
            Column {
                Text("Provider: $provider")
                Text("Expected cost: $costHint", fontWeight = FontWeight.SemiBold)
                Spacer(Modifier.height(8.dp))
                Text(
                    "Billable actions always confirm first. The action itself is not " +
                        "implemented in this slice.",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        },
        confirmButton = {
            TextButton(onClick = onConfirm) { Text("Confirm (stub)") }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text("Cancel") }
        },
    )
}

/** Loss / grad-norm sparkline drawn from real series data. */
@Composable
fun Sparkline(
    values: List<Double>,
    modifier: Modifier = Modifier,
    lineColor: Color = MaterialTheme.colorScheme.primary,
) {
    Canvas(modifier = modifier.fillMaxWidth().height(72.dp)) {
        if (values.size < 2) return@Canvas
        val min = values.min()
        val max = values.max()
        val range = (max - min).takeIf { it > 1e-9 } ?: 1.0
        val path = Path()
        values.forEachIndexed { index, value ->
            val x = size.width * index / (values.size - 1)
            val y = size.height * (1f - ((value - min) / range).toFloat())
            if (index == 0) path.moveTo(x, y) else path.lineTo(x, y)
        }
        drawPath(path, color = lineColor, style = Stroke(width = 3f))
    }
}

/**
 * Heartbeat pulse (arch §10.A): the pulse fires ONLY when [beatTick]
 * changes — i.e. on real tick events from the sim event stream. With no new
 * ticks the trace is flat; there is no decorative animation.
 */
@Composable
fun HeartbeatPulse(
    beatTick: Long,
    modifier: Modifier = Modifier,
    lineColor: Color = MaterialTheme.colorScheme.secondary,
) {
    val amplitude = remember { Animatable(0f) }
    LaunchedEffect(beatTick) {
        if (beatTick > 0) {
            amplitude.snapTo(1f)
            amplitude.animateTo(0f, animationSpec = tween(durationMillis = 600))
        }
    }
    Canvas(modifier = modifier.fillMaxWidth().height(64.dp)) {
        val midY = size.height / 2f
        val amp = amplitude.value * size.height * 0.4f
        val path = Path()
        val spikes = 5
        for (i in 0..spikes) {
            val x = size.width * i / spikes
            val y = if (i % 2 == 1) midY - amp else midY
            if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
        }
        drawPath(path, color = lineColor, style = Stroke(width = 3f))
        drawLine(
            color = lineColor.copy(alpha = 0.3f),
            start = Offset(0f, midY),
            end = Offset(size.width, midY),
            strokeWidth = 1f,
        )
    }
}

/** Simple two-pane container used on expanded (tablet) widths. */
@Composable
fun TwoPane(
    expanded: Boolean,
    first: @Composable () -> Unit,
    second: @Composable () -> Unit,
) {
    if (expanded) {
        Row(Modifier.fillMaxSize()) {
            Box(Modifier.weight(1f).fillMaxSize()) { first() }
            Box(Modifier.weight(1.4f).fillMaxSize()) { second() }
        }
    } else {
        Box(Modifier.fillMaxSize()) { first() }
    }
}

@Composable
fun Chip(text: String, color: Color, modifier: Modifier = Modifier) {
    Surface(
        color = color.copy(alpha = 0.22f),
        shape = MaterialTheme.shapes.small,
        modifier = modifier.padding(end = 4.dp, bottom = 4.dp),
    ) {
        Text(
            text,
            color = color,
            style = MaterialTheme.typography.labelSmall,
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
        )
    }
}
