package com.gliksbot.axonhome.ui.nav

import androidx.compose.runtime.Composable
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewmodel.compose.viewModel

/** Navigation routes (arch §10 screen map). */
object Routes {
    const val HOME = "home"
    const val FIELD = "field"
    const val CORES = "cores"
    const val CORE_DETAIL = "core/{coreId}"
    const val TRAINER = "trainer"
    const val MORE = "more"
    const val COMPUTE = "compute"
    const val STORAGE = "storage"
    const val AGENTS = "agents"
    const val MASKS = "masks"
    const val SETTINGS = "settings"
    const val AUDIT = "audit"
    const val VAULT = "vault"
    // Honest stubs (arch §18 phases)
    const val TIME_MACHINE = "time_machine"
    const val RAIL_OBSERVATORY = "rail_observatory"
    const val SEMANTIC_CORTEX = "semantic_cortex"
    const val SOUL_OBSERVATORY = "soul_observatory"
    const val DORMANT = "dormant"
    const val NOTEBOOK = "notebook"
    const val SSH_TERMINAL = "ssh_terminal"
    const val GIT_CLIENT = "git_client"

    fun coreDetail(coreId: String) = "core/$coreId"
}

/** Create a ViewModel with constructor arguments. */
@Composable
inline fun <reified VM : ViewModel> containerViewModel(
    key: String? = null,
    crossinline create: () -> VM,
): VM {
    val factory = object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>): T = create() as T
    }
    return if (key == null) viewModel(factory = factory) else viewModel(key = key, factory = factory)
}
