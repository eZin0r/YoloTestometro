package br.com.heydjow.yolotestometro

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

class YoloTflite(
    context: Context,
    modelName: String = "model.tflite"
) : AutoCloseable {
    private val interpreter: Interpreter
    private val preprocessor = Preprocessor()
    private val labels = mutableListOf<String>()

    val inputWidth: Int
    val inputHeight: Int
    val inputChannels: Int

    init {
        val model = loadModel(context, modelName)
        interpreter = Interpreter(model)
        val input = interpreter.getInputTensor(0)
        val shape = input.shape()
        require(shape.size == 4) { "Unsupported TFLite input shape: ${shape.contentToString()}" }
        inputHeight = shape[1]
        inputWidth = shape[2]
        inputChannels = shape[3]
        require(inputChannels == 3) { "Expected RGB model input, got $inputChannels channels" }
        loadLabels(context)
    }

    fun detect(
        source: Bitmap,
        confidenceThreshold: Float = 0.25f,
        iouThreshold: Float = 0.45f,
        maxDetections: Int = 100
    ): PipelineMetrics {
        val totalStart = System.nanoTime()
        val preprocessStart = System.nanoTime()
        val prepared = preprocessor.letterbox(source, inputWidth, inputHeight)
        val input = bitmapToBuffer(prepared.bitmap)
        val preprocessMs = elapsedMs(preprocessStart)

        val outputTensor = interpreter.getOutputTensor(0)
        val outputShape = outputTensor.shape()
        val outputBuffer = ByteBuffer.allocateDirect(outputTensor.numElements() * 4).order(ByteOrder.nativeOrder())
        val inferenceStart = System.nanoTime()
        interpreter.run(input, outputBuffer)
        val inferenceMs = elapsedMs(inferenceStart)

        val postprocessStart = System.nanoTime()
        val detections = parseOutput(
            outputBuffer,
            outputShape,
            prepared,
            source.width,
            source.height,
            confidenceThreshold,
            iouThreshold,
            maxDetections
        )
        val postprocessMs = elapsedMs(postprocessStart)
        val totalMs = elapsedMs(totalStart)
        if (!prepared.bitmap.isRecycled) prepared.bitmap.recycle()
        return PipelineMetrics(preprocessMs, inferenceMs, postprocessMs, totalMs, detections)
    }

    fun className(id: Int): String = labels.getOrNull(id) ?: "class_$id"

    private fun bitmapToBuffer(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(inputWidth * inputHeight * 3 * 4).order(ByteOrder.nativeOrder())
        val pixels = IntArray(inputWidth * inputHeight)
        bitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)
        for (pixel in pixels) {
            buffer.putFloat(((pixel shr 16) and 0xff) / 255f)
            buffer.putFloat(((pixel shr 8) and 0xff) / 255f)
            buffer.putFloat((pixel and 0xff) / 255f)
        }
        buffer.rewind()
        return buffer
    }

    private fun parseOutput(
        buffer: ByteBuffer,
        shape: IntArray,
        prepared: Preprocessor.PreparedImage,
        sourceWidth: Int,
        sourceHeight: Int,
        confidenceThreshold: Float,
        iouThreshold: Float,
        maxDetections: Int
    ): List<YoloResult> {
        if (shape.size != 3 || shape[0] != 1) {
            throw IllegalArgumentException("Unsupported YOLO output shape: ${shape.contentToString()}")
        }
        val a = shape[1]
        val b = shape[2]
        val candidates = ArrayList<YoloResult>()
        buffer.rewind()
        val data = FloatArray(a * b)
        buffer.asFloatBuffer().get(data)
        val transposed = a < b
        val attributes = if (transposed) a else b
        val count = if (transposed) b else a
        if (attributes < 5) throw IllegalArgumentException("Unsupported YOLO output attributes: $attributes")
        for (i in 0 until count) {
            fun value(attr: Int): Float = if (transposed) data[attr * count + i] else data[i * b + attr]
            val cx = value(0)
            val cy = value(1)
            val w = value(2)
            val h = value(3)
            var bestClass = -1
            var bestScore = 0f
            for (c in 4 until attributes) {
                val score = value(c)
                if (score > bestScore) {
                    bestScore = score
                    bestClass = c - 4
                }
            }
            if (bestClass < 0 || bestScore < confidenceThreshold) continue
            val leftModel = cx - w / 2f
            val topModel = cy - h / 2f
            val rightModel = cx + w / 2f
            val bottomModel = cy + h / 2f
            val left = ((leftModel - prepared.offsetX) / prepared.scale).coerceIn(0f, sourceWidth.toFloat())
            val top = ((topModel - prepared.offsetY) / prepared.scale).coerceIn(0f, sourceHeight.toFloat())
            val right = ((rightModel - prepared.offsetX) / prepared.scale).coerceIn(0f, sourceWidth.toFloat())
            val bottom = ((bottomModel - prepared.offsetY) / prepared.scale).coerceIn(0f, sourceHeight.toFloat())
            if (right > left && bottom > top) candidates += YoloResult(left, top, right, bottom, bestClass, bestScore)
        }
        return nms(candidates, iouThreshold, maxDetections)
    }

    private fun nms(items: List<YoloResult>, threshold: Float, maxDetections: Int): List<YoloResult> {
        val remaining = items.sortedByDescending { it.confidence }.toMutableList()
        val selected = ArrayList<YoloResult>()
        while (remaining.isNotEmpty() && selected.size < maxDetections) {
            val best = remaining.removeAt(0)
            selected += best
            remaining.removeAll { it.classId == best.classId && iou(best, it) > threshold }
        }
        return selected
    }

    private fun iou(a: YoloResult, b: YoloResult): Float {
        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)
        val intersection = max(0f, right - left) * max(0f, bottom - top)
        val union = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - intersection
        return if (union <= 0f) 0f else intersection / union
    }

    private fun loadModel(context: Context, name: String): ByteBuffer {
        val afd = context.assets.openFd(name)
        FileInputStream(afd.fileDescriptor).use { input ->
            return input.channel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
        }
    }

    private fun loadLabels(context: Context) {
        runCatching {
            context.assets.open("labels.txt").bufferedReader().useLines { lines ->
                labels += lines.map { it.trim() }.filter { it.isNotEmpty() }.toList()
            }
        }
    }

    private fun elapsedMs(start: Long): Double = (System.nanoTime() - start) / 1_000_000.0

    override fun close() {
        interpreter.close()
    }
}
