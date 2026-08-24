package br.com.heydjow.yolotestometro

import android.graphics.Bitmap
import kotlin.math.min

class Preprocessor {
    data class PreparedImage(
        val bitmap: Bitmap,
        val scale: Float,
        val offsetX: Float,
        val offsetY: Float
    )

    fun letterbox(source: Bitmap, inputWidth: Int, inputHeight: Int): PreparedImage {
        require(inputWidth > 0 && inputHeight > 0) { "Invalid model input size" }
        val scale = min(inputWidth.toFloat() / source.width, inputHeight.toFloat() / source.height)
        val resizedWidth = (source.width * scale).toInt().coerceAtLeast(1)
        val resizedHeight = (source.height * scale).toInt().coerceAtLeast(1)
        val output = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(output)
        canvas.drawColor(android.graphics.Color.rgb(114, 114, 114))
        val left = (inputWidth - resizedWidth) / 2f
        val top = (inputHeight - resizedHeight) / 2f
        val resized = Bitmap.createScaledBitmap(source, resizedWidth, resizedHeight, true)
        canvas.drawBitmap(resized, left, top, null)
        if (resized !== source && !resized.isRecycled) resized.recycle()
        return PreparedImage(output, scale, left, top)
    }
}
