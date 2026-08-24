package br.com.heydjow.yolotestometro

import android.content.Context
import java.io.File

class Logger(context: Context) {
    private val file = File(context.filesDir, "yolotestometro.log")

    @Synchronized
    fun append(text: String) {
        file.appendText(text + "\n")
    }

    fun path(): String = file.absolutePath
}
