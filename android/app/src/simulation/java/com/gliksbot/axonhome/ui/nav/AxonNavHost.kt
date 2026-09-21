package com.gliksbot.axonhome.ui.nav

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AccountCircle
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.List
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.windowsizeclass.WindowSizeClass
import androidx.compose.material3.windowsizeclass.WindowWidthSizeClass
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.gliksbot.axonhome.AppContainer
import com.gliksbot.axonhome.ui.agents.AgentsScreen
import com.gliksbot.axonhome.ui.agents.AgentsViewModel
import com.gliksbot.axonhome.ui.audit.AuditScreen
import com.gliksbot.axonhome.ui.components.SimulationBanner
import com.gliksbot.axonhome.ui.components.StubScreen
import com.gliksbot.axonhome.ui.compute.ComputeScreen
import com.gliksbot.axonhome.ui.compute.ComputeViewModel
import com.gliksbot.axonhome.ui.cores.CoreDetailScreen
import com.gliksbot.axonhome.ui.cores.CoreListScreen
import com.gliksbot.axonhome.ui.cores.CoresViewModel
import com.gliksbot.axonhome.ui.field.FieldScreen
import com.gliksbot.axonhome.ui.field.FieldViewModel
import com.gliksbot.axonhome.ui.home.HomeScreen
import com.gliksbot.axonhome.ui.home.HomeViewModel
import com.gliksbot.axonhome.ui.palette.CommandPalette
import com.gliksbot.axonhome.ui.settings.SettingsScreen
import com.gliksbot.axonhome.ui.settings.SettingsViewModel
import com.gliksbot.axonhome.ui.storage.StorageScreen
import com.gliksbot.axonhome.ui.storage.StorageViewModel
import com.gliksbot.axonhome.ui.trainer.TrainerScreen
import com.gliksbot.axonhome.ui.trainer.TrainerViewModel
import com.gliksbot.axonhome.vault.VaultScreen
import com.gliksbot.axonhome.vault.VaultViewModel

private data class TopDestination(val route: String, val label: String)

private val topDestinations = listOf(
    TopDestination(Routes.HOME, "Home"),
    TopDestination(Routes.FIELD, "Field"),
    TopDestination(Routes.CORES, "Cores"),
    TopDestination(Routes.TRAINER, "Trainer"),
    TopDestination(Routes.MORE, "More"),
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AxonNavHost(container: AppContainer, windowSizeClass: WindowSizeClass) {
    val navController = rememberNavController()
    var paletteOpen by remember { mutableStateOf(false) }
    val expanded = windowSizeClass.widthSizeClass == WindowWidthSizeClass.Expanded

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Axon Home") },
                actions = {
                    IconButton(onClick = { paletteOpen = true }) {
                        Icon(Icons.Default.Search, contentDescription = "Command palette")
                    }
                },
            )
        },
        bottomBar = {
            NavigationBar {
                val backStackEntry by navController.currentBackStackEntryAsState()
                val currentRoute = backStackEntry?.destination?.route
                topDestinations.forEach { destination ->
                    NavigationBarItem(
                        selected = currentRoute == destination.route,
                        onClick = {
                            navController.navigate(destination.route) {
                                popUpTo(navController.graph.findStartDestination().id) { saveState = true }
                                launchSingleTop = true
                                restoreState = true
                            }
                        },
                        icon = {
                            Icon(
                                when (destination.route) {
                                    Routes.HOME -> Icons.Default.Home
                                    Routes.FIELD -> Icons.Default.List
                                    Routes.CORES -> Icons.Default.AccountCircle
                                    Routes.TRAINER -> Icons.Default.PlayArrow
                                    else -> Icons.Default.Menu
                                },
                                contentDescription = destination.label,
                            )
                        },
                        label = { Text(destination.label) },
                    )
                }
            }
        },
    ) { padding ->
        Column(Modifier.fillMaxSize().padding(padding)) {
            // Persistent, unmistakable SIMULATION banner on every screen (arch §11).
            SimulationBanner()
            NavHost(
                navController = navController,
                startDestination = Routes.HOME,
                modifier = Modifier.fillMaxSize(),
            ) {
                composable(Routes.HOME) {
                    val vm = containerViewModel { HomeViewModel(container.engine, container.driver) }
                    HomeScreen(vm, onNavigate = { navController.navigate(it) })
                }
                composable(Routes.FIELD) {
                    val vm = containerViewModel { FieldViewModel(container.engine, container.client, container.driver) }
                    FieldScreen(vm, expanded = expanded, onOpenMasks = { navController.navigate(Routes.MASKS) })
                }
                composable(Routes.CORES) {
                    val vm = containerViewModel { CoresViewModel(container.engine, container.client, container.driver) }
                    CoreListScreen(vm, expanded = expanded, onOpenCore = { navController.navigate(Routes.coreDetail(it)) })
                }
                composable(
                    Routes.CORE_DETAIL,
                    arguments = listOf(navArgument("coreId") { type = NavType.StringType }),
                ) { entry ->
                    val coreId = entry.arguments?.getString("coreId").orEmpty()
                    val vm = containerViewModel(key = "core-$coreId") { CoresViewModel(container.engine, container.client, container.driver) }
                    CoreDetailScreen(vm, coreId = coreId, onBack = { navController.popBackStack() })
                }
                composable(Routes.TRAINER) {
                    val vm = containerViewModel { TrainerViewModel(container.engine, container.client, container.driver) }
                    TrainerScreen(vm)
                }
                composable(Routes.MORE) {
                    MoreScreen(onNavigate = { navController.navigate(it) })
                }
                composable(Routes.COMPUTE) {
                    val vm = containerViewModel { ComputeViewModel(container.engine, container.client, container.profiles, container.driver, { c, t, p2, r, res -> container.audit(c, t, p2, r, res) }) }
                    ComputeScreen(vm)
                }
                composable(Routes.STORAGE) {
                    val vm = containerViewModel { StorageViewModel(container.engine, container.client, container.driver) }
                    StorageScreen(vm)
                }
                composable(Routes.AGENTS) {
                    val vm = containerViewModel { AgentsViewModel(container.engine, container.client, container.profiles, container.driver, { c, t, p2, r, res -> container.audit(c, t, p2, r, res) }) }
                    AgentsScreen(vm)
                }
                composable(Routes.MASKS) {
                    val vm = containerViewModel { FieldViewModel(container.engine, container.client, container.driver) }
                    com.gliksbot.axonhome.ui.field.MaskScreen(vm, onBack = { navController.popBackStack() })
                }
                composable(Routes.SETTINGS) {
                    val vm = containerViewModel { SettingsViewModel(container) }
                    SettingsScreen(vm, onOpenAudit = { navController.navigate(Routes.AUDIT) })
                }
                composable(Routes.AUDIT) {
                    AuditScreen(container)
                }
                composable(Routes.VAULT) {
                    val vm = containerViewModel { VaultViewModel(container) }
                    VaultScreen(vm)
                }
                composable(Routes.TIME_MACHINE) {
                    StubScreen(
                        "Tick Time Machine — deep inspector",
                        "Phase 6 (arch §18)",
                        "Basic tick navigation lives in the Field Explorer. The deep " +
                            "per-phase inspector (frozen images, per-core timing, soul " +
                            "receipt drill-down) is designed, not built.",
                    )
                }
                composable(Routes.RAIL_OBSERVATORY) {
                    StubScreen(
                        "Rail Observatory",
                        "Phase 6 (arch §18)",
                        "D16 substrate packing, page traversal and attended-interval " +
                            "coverage require the field compiler manifests, which the " +
                            "simulator does not emit.",
                    )
                }
                composable(Routes.SEMANTIC_CORTEX) {
                    StubScreen(
                        "Semantic Cortex",
                        "Phase 6 (arch §18)",
                        "Cortext is reserved/inactive in the organism; there is nothing " +
                            "truthful to render yet.",
                    )
                }
                composable(Routes.SOUL_OBSERVATORY) {
                    StubScreen(
                        "Soul Observatory — deep view",
                        "Phase 6 (arch §18)",
                        "Per-core soul id, generation and receipt chains are real in Core " +
                            "detail. Causal probes and migrations are designed, not built. " +
                            "Latent tensors are opaque and never rendered as semantics.",
                    )
                }
                composable(Routes.DORMANT) {
                    DormantScreen(container)
                }
                composable(Routes.NOTEBOOK) {
                    StubScreen(
                        "Notebook / Remote Workspace",
                        "Phase 4 (arch §18)",
                        "Kaggle/Colab/Jupyter status and terminal views need live provider " +
                            "adapters; no network calls exist in this build.",
                    )
                }
                composable(Routes.SSH_TERMINAL) {
                    StubScreen(
                        "SSH Terminal",
                        "Phase 4 (arch §18)",
                        "Vault-backed keys and host fingerprint TOFU are designed (arch §8); " +
                            "the terminal transport is not built.",
                    )
                }
                composable(Routes.GIT_CLIENT) {
                    GitScreen(container)
                }
            }
        }
    }

    if (paletteOpen) {
        CommandPalette(
            container = container,
            onDismiss = { paletteOpen = false },
            onNavigate = { route ->
                paletteOpen = false
                navController.navigate(route) { launchSingleTop = true }
            },
        )
    }
}
