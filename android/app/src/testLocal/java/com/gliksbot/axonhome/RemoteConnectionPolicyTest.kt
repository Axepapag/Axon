package com.gliksbot.axonhome

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class RemoteConnectionPolicyTest {
    @Test
    fun cleartextIsRestrictedToExactLocalhost() {
        assertTrue(isAllowedControlEndpoint("http://localhost:8000"))
        assertFalse(isAllowedControlEndpoint("http://127.0.0.1:8000"))
        assertFalse(isAllowedControlEndpoint("http://example.com"))
        assertFalse(isAllowedControlEndpoint("http://localhost.example.com"))
    }

    @Test
    fun remoteHostsRequireHttpsAndRejectAmbiguousUrls() {
        assertTrue(isAllowedControlEndpoint("https://axon.example.com"))
        assertFalse(isAllowedControlEndpoint("https://user@axon.example.com"))
        assertFalse(isAllowedControlEndpoint("https://axon.example.com?token=leak"))
    }
}
