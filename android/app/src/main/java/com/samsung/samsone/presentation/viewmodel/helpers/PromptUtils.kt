package com.samsung.samsone.presentation.viewmodel.helpers

/**
 * Prompt normalization matching SamsoneModel._normalise_prompt on the Python side
 * (and the text preprocessing used in training). The pruned vocabulary only has
 * lowercase ASCII tokens, so raw user input would otherwise produce token ids the
 * model does not have.
 */
object PromptUtils {
    private val nonAscii = Regex("[^\\x00-\\x7F]")

    // Python's \s also matches \x1c-\x1f
    private val longWhitespace = Regex("[\\s\\x1C-\\x1F]{4,}")
    private val dashes = Regex("-{3,}")
    private val hashes = Regex("#{3,}")

    fun normalize(prompt: String): String =
        prompt.lowercase()
            .replace(nonAscii, "")
            .replace(longWhitespace, " ")
            .replace(dashes, "-")
            .replace(hashes, "#")
}
