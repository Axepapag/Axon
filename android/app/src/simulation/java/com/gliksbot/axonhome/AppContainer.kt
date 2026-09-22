package com.gliksbot.axonhome

import android.content.Context
import com.gliksbot.axonhome.data.ProfileStore
import com.gliksbot.axonhome.data.SettingsStore
import com.gliksbot.axonhome.sim.SimControlPlaneClient
import com.gliksbot.axonhome.sim.SimulationDriver
import com.gliksbot.axonhome.core.sim.SimulationEngine
import com.gliksbot.axonhome.vault.VaultStore
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.datetime.Clock

/**
 * Manual DI graph. v1 wiring: the deterministic simulator behind the
 * ControlPlaneClient contract (arch §11). Production wiring swaps
 * [SimControlPlaneClient] for a networked implementation — no UI changes.
 */
class AppContainer(context: Context) {

    val appScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    val engine = SimulationEngine()
    val client = SimControlPlaneClient(engine)
    val driver = SimulationDriver(engine, appScope)

    val settings = SettingsStore(context)
    val profiles = ProfileStore(context)
    val vault = VaultStore(context)

    /** Global "lock vault now" signal (command palette → vault). */
    val vaultLockSignal = kotlinx.coroutines.flow.MutableStateFlow(0L)
    fun lockVault() {
        vaultLockSignal.value += 1
        audit("vault_lock", "vault", "unlocked", "locked", "locked")
    }

    init {
        // Seed a few heartbeats so the first frame renders real organism state.
        repeat(3) { driver.stepOnce() }
        driver.start()
        appScope.launch { settings.tickIntervalMs.collect { driver.tickIntervalMs.value = it } }
        appScope.launch { settings.autoTick.collect { driver.autoTick.value = it } }
    }

    /** Record an app-initiated control action in the :core audit log (arch §14). */
    fun audit(command: String, target: String, previous: String, requested: String, result: String, error: String? = null) {
        engine.auditLog.record(
            who = "operator",
            device = "android-app",
            command = command,
            target = target,
            timestamp = Clock.System.now().toString(),
            previousState = previous,
            requestedState = requested,
            result = result,
            error = error,
        )
    }
}
