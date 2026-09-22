package com.gliksbot.axonhome.core.domain

import kotlinx.serialization.json.Json

/**
 * Shared wire codec for :core. Schema strings, region names and field names
 * match the canonical Axon JSON contracts exactly (snake_case via
 * @SerialName on every model). Defaults and explicit nulls are encoded so a
 * round trip is stable; unknown keys are ignored for forward compatibility.
 */
val AxonJson: Json = Json {
    encodeDefaults = true
    explicitNulls = true
    ignoreUnknownKeys = true
    prettyPrint = false
}
