package com.samsung.samsone.presentation.viewmodel.helpers

import org.junit.Assert.assertEquals
import org.junit.Test

class PromptUtilsTest {
    // expected values produced by SamsoneModel._normalise_prompt (src/samsone/inference/models.py)
    private val cases = listOf(
        "What is this sound?" to "what is this sound?",
        "Describe the audio." to "describe the audio.",
        "what’s happening here?" to "whats happening here?",
        "Is it a DOG or a cat?" to "is it a dog or a cat?",
        "one    two\t\t\t\tthree" to "one two three",
        "wait---what ###now" to "wait-what #now",
        "café naïve über" to "caf nave ber",
        "  keep  three   spaces  " to "  keep  three   spaces  ",
    )

    @Test
    fun matchesPythonNormalization() {
        for ((input, expected) in cases) {
            assertEquals(input, expected, PromptUtils.normalize(input))
        }
    }
}
