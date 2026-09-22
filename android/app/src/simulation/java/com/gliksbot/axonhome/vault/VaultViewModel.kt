package com.gliksbot.axonhome.vault

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.gliksbot.axonhome.AppContainer
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch

data class VaultState(
    val locked: Boolean = true,
    val entries: List<VaultEntryMeta> = emptyList(),
    val feedback: String? = null,
    /** Entry id awaiting re-auth before reveal/copy. */
    val pendingRevealId: String? = null,
)

/**
 * Secrets Vault (foundation scope, arch §8): Keystore-sealed entries,
 * masked display, re-auth reveal, auto-lock. Never logs secrets.
 */
class VaultViewModel(private val container: AppContainer) : ViewModel() {

    private val _state = MutableStateFlow(VaultState())
    val state: StateFlow<VaultState> = _state

    private var lockJob: Job? = null

    init {
        viewModelScope.launch { refreshEntries() }
        viewModelScope.launch {
            var seen = container.vaultLockSignal.value
            container.vaultLockSignal.collect { signal ->
                if (signal != seen) {
                    seen = signal
                    lock()
                }
            }
        }
    }

    private suspend fun refreshEntries() {
        _state.value = _state.value.copy(entries = container.vault.entries())
    }

    /** Called only after a successful biometric/device-auth prompt (or the
     *  explicit fallback confirm when no authenticator exists). */
    fun onUnlockSuccess() {
        lockJob?.cancel()
        _state.value = _state.value.copy(locked = false, feedback = null)
        viewModelScope.launch { refreshEntries() }
    }

    fun lock() {
        _state.value = _state.value.copy(locked = true, pendingRevealId = null)
    }

    /** App backgrounded: lock now or after the configured timeout (reads Settings). */
    fun onBackgroundAuto() {
        lockJob?.cancel()
        lockJob = viewModelScope.launch {
            val seconds = runCatching { container.settings.autoLockSeconds.first() }.getOrDefault(60)
            if (seconds <= 0) lock() else {
                delay(seconds * 1000L)
                lock()
            }
        }
    }

    /** App backgrounded with explicit timeout. */
    fun onBackground(autoLockSeconds: Int) {
        lockJob?.cancel()
        lockJob = viewModelScope.launch {
            if (autoLockSeconds <= 0) {
                lock()
            } else {
                delay(autoLockSeconds * 1000L)
                lock()
            }
        }
    }

    fun addSecret(label: String, scope: String, secret: String) {
        if (label.isBlank() || secret.isEmpty()) return
        viewModelScope.launch {
            container.vault.add(label.trim(), scope.trim().ifEmpty { "general" }, secret)
            container.audit("vault_add", "vault", "n/a", label.trim(), "sealed")
            _state.value = _state.value.copy(feedback = "Secret sealed. It is never shown in full again.")
            refreshEntries()
        }
    }

    fun deleteSecret(id: String) {
        viewModelScope.launch {
            container.vault.delete(id)
            container.audit("vault_delete", "vault", "sealed", "absent", "deleted")
            refreshEntries()
        }
    }

    /** Begin a reveal: the UI must re-authenticate before unseal succeeds. */
    fun requestReveal(id: String) {
        _state.value = _state.value.copy(pendingRevealId = id)
    }

    fun cancelReveal() {
        _state.value = _state.value.copy(pendingRevealId = null)
    }

    /** Unseal after re-auth; the caller copies to clipboard flagged sensitive. */
    suspend fun unsealAfterAuth(id: String): String? {
        if (_state.value.locked) return null
        if (_state.value.pendingRevealId != id) return null
        _state.value = _state.value.copy(pendingRevealId = null)
        container.audit("vault_reveal", "vault", "sealed", id, "revealed-masked-copy")
        return container.vault.unseal(id)
    }
}
