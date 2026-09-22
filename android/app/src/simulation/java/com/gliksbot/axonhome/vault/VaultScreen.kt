package com.gliksbot.axonhome.vault

import android.content.ClipData
import android.content.ClipDescription
import android.content.Context
import android.os.Build
import android.os.PersistableBundle
import androidx.biometric.BiometricManager
import androidx.biometric.BiometricPrompt
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
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.fragment.app.FragmentActivity
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.gliksbot.axonhome.data.SettingsStore
import com.gliksbot.axonhome.ui.components.InfoRow
import com.gliksbot.axonhome.ui.components.SectionCard
import com.gliksbot.axonhome.ui.theme.AxonError
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch

private fun deviceAuthAvailable(context: Context): Boolean =
    BiometricManager.from(context).canAuthenticate(
        BiometricManager.Authenticators.BIOMETRIC_WEAK or BiometricManager.Authenticators.DEVICE_CREDENTIAL,
    ) == BiometricManager.BIOMETRIC_SUCCESS

private fun promptDeviceAuth(activity: FragmentActivity, onSuccess: () -> Unit, onError: (String) -> Unit) {
    val executor = ContextCompat.getMainExecutor(activity)
    val prompt = BiometricPrompt(
        activity,
        executor,
        object : BiometricPrompt.AuthenticationCallback() {
            override fun onAuthenticationSucceeded(result: BiometricPrompt.AuthenticationResult) = onSuccess()
            override fun onAuthenticationError(code: Int, message: CharSequence) = onError(message.toString())
        },
    )
    val info = BiometricPrompt.PromptInfo.Builder()
        .setTitle("Axon Home Vault")
        .setSubtitle("Authenticate to continue")
        .setAllowedAuthenticators(
            BiometricManager.Authenticators.BIOMETRIC_WEAK or BiometricManager.Authenticators.DEVICE_CREDENTIAL,
        )
        .build()
    prompt.authenticate(info)
}

/** Copy a secret flagged sensitive; the system clipboard is the user's responsibility. */
private fun copySecret(context: Context, label: String, value: String) {
    val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as android.content.ClipboardManager
    val clip = ClipData.newPlainText(label, value)
    if (Build.VERSION.SDK_INT >= 33) {
        clip.description.extras = PersistableBundle().apply {
            putBoolean(ClipDescription.EXTRA_IS_SENSITIVE, true)
        }
    } else if (Build.VERSION.SDK_INT >= 24) {
        clip.description.extras = PersistableBundle().apply { putBoolean("android.content.extra.IS_SENSITIVE", true) }
    }
    clipboard.setPrimaryClip(clip)
}

@Composable
fun VaultScreen(viewModel: VaultViewModel) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val lifecycleOwner = LocalLifecycleOwner.current
    var feedback by remember { mutableStateOf<String?>(null) }

    // Auto-lock on background after the configured timeout (arch §8).
    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_STOP) viewModel.onBackgroundAuto()
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Text("Secrets Vault", style = MaterialTheme.typography.titleLarge)
        Text(
            "Foundation scope (arch §8): AES-GCM keys in AndroidKeyStore, sealed " +
                "per entry; values are never shown in full after entry.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        if (state.locked) {
            SectionCard("Locked") {
                Text("The vault is locked. Authenticate to view entries.")
                Button(onClick = {
                    val activity = context as? FragmentActivity
                    if (activity == null || !deviceAuthAvailable(context)) {
                        feedback = "No biometric/device credential enrolled — unlocking without auth (degraded, dev only)."
                        viewModel.onUnlockSuccess()
                    } else {
                        promptDeviceAuth(
                            activity,
                            onSuccess = { viewModel.onUnlockSuccess() },
                            onError = { feedback = "Auth failed: $it" },
                        )
                    }
                }, modifier = Modifier.padding(top = 8.dp)) { Text("Unlock") }
            }
        } else {
            SectionCard("Entries (${state.entries.size})") {
                state.entries.forEach { entry ->
                    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(vertical = 4.dp)) {
                        Column(Modifier.weight(1f)) {
                            Text(entry.label, style = MaterialTheme.typography.bodyMedium)
                            Text(
                                "${entry.scope} · ${entry.maskedHint}",
                                style = MaterialTheme.typography.bodySmall,
                                fontFamily = FontFamily.Monospace,
                            )
                        }
                        OutlinedButton(onClick = { viewModel.requestReveal(entry.id) }) { Text("Copy") }
                        OutlinedButton(
                            onClick = { viewModel.deleteSecret(entry.id) },
                            Modifier.padding(start = 4.dp),
                        ) { Text("Delete") }
                    }
                }
                if (state.entries.isEmpty()) Text("No secrets yet. Each entry is scoped to one provider — least privilege by default.")
            }

            AddSecretCard(onAdd = { label, scope, secret -> viewModel.addSecret(label, scope, secret) })

            OutlinedButton(onClick = { viewModel.lock() }) { Text("Lock now") }
        }

        (feedback ?: state.feedback)?.let {
            Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.secondary)
        }
    }

    // Re-auth before unseal/copy.
    state.pendingRevealId?.let { revealId ->
        val entry = state.entries.firstOrNull { it.id == revealId }
        if (entry != null) {
            AlertDialog(
                onDismissRequest = { viewModel.cancelReveal() },
                title = { Text("Reveal ${entry.label}?") },
                text = {
                    Text(
                        "Re-authentication is required. The value is copied to the " +
                            "clipboard flagged sensitive and is NEVER displayed in full. " +
                            "Clear your clipboard after use.",
                    )
                },
                confirmButton = {
                    Button(onClick = {
                        val activity = context as? FragmentActivity
                        val doCopy: () -> Unit = {
                            scope.launch {
                                val value = viewModel.unsealAfterAuth(revealId)
                                if (value != null) {
                                    copySecret(context, entry.label, value)
                                    feedback = "Copied '${entry.label}' — flagged sensitive; auto-clear is the OS's job, clear it after use."
                                } else {
                                    feedback = "Reveal refused (vault locked or request stale)."
                                }
                            }
                        }
                        if (activity == null || !deviceAuthAvailable(context)) {
                            feedback = "No authenticator available — proceeding without re-auth (degraded, dev only)."
                            doCopy()
                        } else {
                            promptDeviceAuth(activity, onSuccess = doCopy, onError = { feedback = "Auth failed: $it" })
                        }
                    }) { Text("Authenticate & copy") }
                },
                dismissButton = {
                    OutlinedButton(onClick = { viewModel.cancelReveal() }) { Text("Cancel") }
                },
            )
        }
    }
}

@Composable
private fun AddSecretCard(onAdd: (String, String, String) -> Unit) {
    var label by remember { mutableStateOf("") }
    var scope by remember { mutableStateOf("") }
    var secret by remember { mutableStateOf("") }

    SectionCard("Add secret") {
        OutlinedTextField(label, { label = it }, label = { Text("Label") }, modifier = Modifier.fillMaxWidth())
        OutlinedTextField(scope, { scope = it }, label = { Text("Scope (e.g. github, kaggle, ssh)") }, modifier = Modifier.fillMaxWidth())
        OutlinedTextField(
            secret, { secret = it },
            label = { Text("Secret value") },
            modifier = Modifier.fillMaxWidth(),
        )
        Button(
            onClick = { onAdd(label, scope, secret); label = ""; scope = ""; secret = "" },
            enabled = label.isNotBlank() && secret.isNotEmpty(),
            modifier = Modifier.padding(top = 6.dp),
        ) { Text("Seal into vault") }
        Text(
            "After sealing, only the last 4 characters are ever shown.",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}
