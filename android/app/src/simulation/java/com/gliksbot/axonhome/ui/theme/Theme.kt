package com.gliksbot.axonhome.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import com.gliksbot.axonhome.data.ThemeMode

// Dark-first palette (arch §4.1). Simulation accent: a hard-to-miss amber.
val SimAccent = Color(0xFFFFB300)
val AxonPrimary = Color(0xFF7C9EFF)
val AxonSecondary = Color(0xFF66E0C2)
val AxonError = Color(0xFFFF6E6E)

private val DarkColors = darkColorScheme(
    primary = AxonPrimary,
    secondary = AxonSecondary,
    tertiary = SimAccent,
    background = Color(0xFF0B0E14),
    surface = Color(0xFF121722),
    surfaceVariant = Color(0xFF1B2230),
    error = AxonError,
)

private val LightColors = lightColorScheme(
    primary = Color(0xFF3455B0),
    secondary = Color(0xFF00705A),
    tertiary = Color(0xFF8A6100),
)

@Composable
fun AxonHomeTheme(mode: ThemeMode = ThemeMode.SYSTEM, content: @Composable () -> Unit) {
    val dark = when (mode) {
        ThemeMode.SYSTEM -> isSystemInDarkTheme()
        ThemeMode.LIGHT -> false
        ThemeMode.DARK -> true
    }
    MaterialTheme(
        colorScheme = if (dark) DarkColors else LightColors,
        content = content,
    )
}
