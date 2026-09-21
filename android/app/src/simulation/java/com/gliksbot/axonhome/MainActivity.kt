package com.gliksbot.axonhome

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material3.windowsizeclass.ExperimentalMaterial3WindowSizeClassApi
import androidx.compose.material3.windowsizeclass.calculateWindowSizeClass
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.gliksbot.axonhome.data.ThemeMode
import com.gliksbot.axonhome.ui.nav.AxonNavHost
import com.gliksbot.axonhome.ui.theme.AxonHomeTheme

class MainActivity : ComponentActivity() {

    private lateinit var container: AppContainer

    @OptIn(ExperimentalMaterial3WindowSizeClassApi::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        container = AppContainer(applicationContext)
        setContent {
            val themeMode by container.settings.themeMode
                .collectAsStateWithLifecycle(initialValue = ThemeMode.SYSTEM)
            val windowSizeClass = calculateWindowSizeClass(this)
            AxonHomeTheme(themeMode) {
                AxonNavHost(container, windowSizeClass)
            }
        }
    }

    override fun onStop() {
        super.onStop()
        // Vault auto-lock is handled inside the vault ViewModel via lifecycle
        // observation; the driver keeps ticking while the app is backgrounded
        // so cached state stays visibly labelled.
    }
}
