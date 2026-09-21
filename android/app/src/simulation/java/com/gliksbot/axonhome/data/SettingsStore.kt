package com.gliksbot.axonhome.data

import android.content.Context
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.longPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.dataStore by preferencesDataStore(name = "axon_home_settings")

enum class ThemeMode { SYSTEM, LIGHT, DARK }

/** User preferences (non-secret): theme, refresh cadence, security knobs. */
class SettingsStore(private val context: Context) {

    private val themeKey = stringPreferencesKey("theme_mode")
    private val tickIntervalKey = longPreferencesKey("tick_interval_ms")
    private val autoTickKey = booleanPreferencesKey("auto_tick")
    private val autoLockKey = intPreferencesKey("auto_lock_seconds")
    private val biometricKey = booleanPreferencesKey("biometric_unlock")

    val themeMode: Flow<ThemeMode> = context.dataStore.data.map { prefs ->
        prefs[themeKey]?.let { runCatching { ThemeMode.valueOf(it) }.getOrNull() } ?: ThemeMode.SYSTEM
    }

    val tickIntervalMs: Flow<Long> = context.dataStore.data.map { prefs -> prefs[tickIntervalKey] ?: 2_000L }

    val autoTick: Flow<Boolean> = context.dataStore.data.map { prefs -> prefs[autoTickKey] ?: true }

    /** Auto-lock timeout for the vault, seconds. 0 = lock immediately on background. */
    val autoLockSeconds: Flow<Int> = context.dataStore.data.map { prefs -> prefs[autoLockKey] ?: 60 }

    val biometricEnabled: Flow<Boolean> = context.dataStore.data.map { prefs -> prefs[biometricKey] ?: true }

    suspend fun setThemeMode(mode: ThemeMode) {
        context.dataStore.edit { it[themeKey] = mode.name }
    }

    suspend fun setTickIntervalMs(value: Long) {
        context.dataStore.edit { it[tickIntervalKey] = value.coerceIn(500L, 60_000L) }
    }

    suspend fun setAutoTick(value: Boolean) {
        context.dataStore.edit { it[autoTickKey] = value }
    }

    suspend fun setAutoLockSeconds(value: Int) {
        context.dataStore.edit { it[autoLockKey] = value.coerceIn(0, 3600) }
    }

    suspend fun setBiometricEnabled(value: Boolean) {
        context.dataStore.edit { it[biometricKey] = value }
    }
}
