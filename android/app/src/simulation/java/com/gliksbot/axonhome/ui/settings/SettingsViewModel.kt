package com.gliksbot.axonhome.ui.settings

import android.content.Context
import android.content.Intent
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.gliksbot.axonhome.AppContainer
import com.gliksbot.axonhome.BuildConfig
import com.gliksbot.axonhome.data.ThemeMode
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.launch
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

data class SettingsState(
    val themeMode: ThemeMode = ThemeMode.SYSTEM,
    val tickIntervalMs: Long = 2_000L,
    val autoTick: Boolean = true,
    val autoLockSeconds: Int = 60,
    val biometricEnabled: Boolean = true,
    val endpointCount: Int = 0,
    val feedback: String? = null,
)

/** Non-secret config export payload (arch §10.T). Secrets never leave the vault. */
@Serializable
data class ConfigExport(
    val app: String = "axon-home",
    val version: String,
    val gitCommit: String,
    val themeMode: String,
    val tickIntervalMs: Long,
    val autoLockSeconds: Int,
    val simulation: Boolean = true,
)

class SettingsViewModel(private val container: AppContainer) : ViewModel() {

    private val _state = MutableStateFlow(SettingsState())
    val state: StateFlow<SettingsState> = _state

    init {
        viewModelScope.launch {
            combine(
                container.settings.themeMode,
                container.settings.tickIntervalMs,
                container.settings.autoTick,
                container.settings.autoLockSeconds,
                container.settings.biometricEnabled,
            ) { values ->
                SettingsState(
                    themeMode = values[0] as ThemeMode,
                    tickIntervalMs = values[1] as Long,
                    autoTick = values[2] as Boolean,
                    autoLockSeconds = values[3] as Int,
                    biometricEnabled = values[4] as Boolean,
                )
            }.collect { value ->
                _state.value = _state.value.copy(
                    themeMode = value.themeMode,
                    tickIntervalMs = value.tickIntervalMs,
                    autoTick = value.autoTick,
                    autoLockSeconds = value.autoLockSeconds,
                    biometricEnabled = value.biometricEnabled,
                )
            }
        }
    }

    fun setTheme(mode: ThemeMode) {
        viewModelScope.launch { container.settings.setThemeMode(mode) }
    }

    fun setTickInterval(ms: Long) {
        viewModelScope.launch { container.settings.setTickIntervalMs(ms) }
    }

    fun setAutoTick(value: Boolean) {
        viewModelScope.launch { container.settings.setAutoTick(value) }
    }

    fun setAutoLock(seconds: Int) {
        viewModelScope.launch {
            container.settings.setAutoLockSeconds(seconds)
            container.audit("set_auto_lock", "vault", "n/a", "${seconds}s", "saved")
        }
    }

    fun setBiometric(enabled: Boolean) {
        viewModelScope.launch {
            container.settings.setBiometricEnabled(enabled)
            container.audit("set_biometric_unlock", "vault", "n/a", enabled.toString(), "saved")
        }
    }

    /** Export non-secret configuration as a JSON share intent. */
    fun exportConfig(context: Context) {
        val export = ConfigExport(
            version = BuildConfig.VERSION_NAME,
            gitCommit = BuildConfig.GIT_COMMIT,
            themeMode = _state.value.themeMode.name,
            tickIntervalMs = _state.value.tickIntervalMs,
            autoLockSeconds = _state.value.autoLockSeconds,
        )
        val text = Json { prettyPrint = true }.encodeToString(ConfigExport.serializer(), export)
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "application/json"
            putExtra(Intent.EXTRA_TEXT, text)
            putExtra(Intent.EXTRA_SUBJECT, "Axon Home config (non-secret)")
        }
        context.startActivity(Intent.createChooser(intent, "Export config"))
        container.audit("export_config", "settings", "local", "shared", "exported-non-secret")
    }
}
