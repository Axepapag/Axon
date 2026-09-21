package com.gliksbot.axonhome.phone

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.gliksbot.axonhome.BuildConfig
import com.gliksbot.axonhome.core.domain.*
import com.gliksbot.axonhome.core.phone.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@Composable
fun PhoneHome() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val store = remember { PhoneStore(context.applicationContext) }
    var state by remember { mutableStateOf<PhoneState?>(null) }
    var message by remember { mutableStateOf("Opening local state…") }
    var text by remember { mutableStateOf("") }
    var busy by remember { mutableStateOf(false) }
    var pendingRestore by remember { mutableStateOf<PhoneRecovery?>(null) }
    var selected by remember { mutableStateOf(LogicalRegion.USER_INPUT) }
    fun run(action: suspend () -> String) {
        if (busy) return
        busy = true
        scope.launch {
            try {
                val result = withContext(Dispatchers.IO) { action() }
                state = withContext(Dispatchers.IO) { store.state() }
                message = result
            } catch (e: Exception) { message = "Failed: ${e.message}" }
            finally { busy = false }
        }
    }
    LaunchedEffect(Unit) { run { "Local state loaded. Core inference is unavailable." } }
    val export = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/json")) { uri ->
        if (uri != null) run {
            val capsule = PhoneRecovery.create(store.state(), BuildConfig.GIT_COMMIT)
            val bytes = AxonJson.encodeToString(PhoneRecovery.serializer(), capsule).encodeToByteArray()
            context.contentResolver.openOutputStream(uri, "wt")?.use { it.write(bytes); it.flush() }
                ?: error("Provider did not open the destination")
            val saved = context.contentResolver.openInputStream(uri)?.bufferedReader()?.use { it.readText() }
                ?: error("Cannot verify the exported file")
            val readback = AxonJson.decodeFromString(PhoneRecovery.serializer(), saved)
            readback.verify()
            check(readback.sha256 == capsule.sha256) { "Readback differs from exported body" }
            "Body backup written and read back. Cloud sync depends on the selected file provider."
        }
    }
    val restore = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            busy = true
            scope.launch {
                try {
                    pendingRestore = withContext(Dispatchers.IO) {
                        val data = context.contentResolver.openInputStream(uri)?.bufferedReader()?.use { it.readText() }
                            ?: error("Provider did not open the backup")
                        AxonJson.decodeFromString(PhoneRecovery.serializer(), data).also { it.verify() }
                    }
                    message = "Backup verified. Review restore confirmation."
                } catch (e: Exception) { message = "Restore rejected: ${e.message}" }
                finally { busy = false }
            }
        }
    }
    MaterialTheme(colorScheme = darkColorScheme()) {
        Surface(Modifier.fillMaxSize()) {
            Column(Modifier.safeDrawingPadding().verticalScroll(rememberScrollState()).padding(20.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text("Axon Home", style = MaterialTheme.typography.headlineLarge)
                Text("LOCAL FOUNDATION", color = MaterialTheme.colorScheme.primary)
                Text("Phone state • ${BuildConfig.VERSION_NAME}")
                Text(message)
                if (busy) LinearProgressIndicator(Modifier.fillMaxWidth())
                state?.let { current ->
                    Text(if (current.paused) "Heart ingress: PAUSED" else "Heart ingress: READY")
                    Text("Persisted inputs: ${current.ingressCount}   •   Reasoning ticks: ${current.field.tickId}")
                    Text("Core inference: UNAVAILABLE — no accepted mobile Core installed.")
                    Text("Cloud trainer: DISCONNECTED — no worker has been dispatched.")
                    Text("Field: ${current.field.fieldId}", style = MaterialTheme.typography.bodySmall)
                    OutlinedTextField(value = text, onValueChange = { text = it },
                        label = { Text("Message for Axon") }, modifier = Modifier.fillMaxWidth(), minLines = 3)
                    Button(enabled = !busy && !current.paused && text.isNotBlank(), onClick = {
                        val input = text
                        run { store.ingress(input); "Input persisted. Awaiting an available reasoning Core." }
                    }) { Text("Save input to Heart") }
                    OutlinedButton(enabled = !busy, onClick = { run {
                        store.pause(!current.paused)
                        if (current.paused) "Local ingress resumed." else "Local ingress paused. No remote worker is controlled by this button."
                    } }) { Text(if (current.paused) "Resume local ingress" else "Pause local ingress") }
                    HorizontalDivider()
                    Text("Shared Field", style = MaterialTheme.typography.titleLarge)
                    var expanded by remember { mutableStateOf(false) }
                    Box {
                        OutlinedButton(onClick = { expanded = true }) { Text(selected.wireName) }
                        DropdownMenu(expanded, onDismissRequest = { expanded = false }) {
                            CANONICAL_REGION_ORDER.forEach { region ->
                                DropdownMenuItem(text = { Text(region.wireName) }, onClick = { selected = region; expanded = false })
                            }
                        }
                    }
                    val percent = current.masks.getValue(selected.wireName)
                    var slider by remember(selected, percent) { mutableFloatStateOf(percent.toFloat()) }
                    Text("Attended: ${slider.toInt()}%")
                    Slider(value = slider, onValueChange = { slider = it }, valueRange = 0f..100f,
                        enabled = !busy && selected != LogicalRegion.IDENTITY,
                        onValueChangeFinished = { val p = slider.toInt(); run { store.mask(selected, p); "Mask saved; canonical text retained." } })
                    val region = current.field.region(selected)
                    Text("Canonical characters: ${region.text.codePointCount(0, region.text.length)}")
                    Text(region.copy(maskPolicy = RegionMaskPolicy.tailPercent(percent)).attendedText.ifEmpty { "(No attended text)" })
                    HorizontalDivider()
                    Text("Recovery", style = MaterialTheme.typography.titleLarge)
                    Text("Back up this phone's body and masks. Choose Google Drive in the file picker if available. This does not contain Core weights or Souls.")
                    OutlinedButton(enabled = !busy, onClick = { export.launch("axon-phone-body-${System.currentTimeMillis()}.json") }) { Text("Export body backup") }
                    OutlinedButton(enabled = !busy, onClick = { restore.launch(arrayOf("application/json", "application/octet-stream")) }) { Text("Restore body backup") }
                }
            }
        }
        pendingRestore?.let { capsule ->
            AlertDialog(onDismissRequest = { pendingRestore = null },
                title = { Text("Restore this phone body?") },
                text = { Text("This changes the active body and masks to the verified backup. Your previous body remains in local transaction history. Core weights and Souls are not included.") },
                confirmButton = { TextButton(onClick = {
                    pendingRestore = null
                    run { store.restore(capsule); "Body restored and committed locally." }
                }) { Text("Restore") } },
                dismissButton = { TextButton(onClick = { pendingRestore = null }) { Text("Cancel") } })
        }
    }
}
