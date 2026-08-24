package br.com.heydjow.yolotestometro

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.view.View
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import java.util.Locale
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private val executor = Executors.newSingleThreadExecutor()
    private lateinit var yolo: YoloTflite
    private lateinit var logger: Logger
    private var selected: Uri? = null
    private var currentBitmap: Bitmap? = null
    private var lastExportBitmap: Bitmap? = null
    private lateinit var status: TextView
    private lateinit var details: TextView
    private lateinit var inputWidth: EditText
    private lateinit var inputHeight: EditText
    private lateinit var confidence: EditText
    private lateinit var iou: EditText
    private lateinit var warmup: EditText
    private lateinit var repetitions: EditText
    private lateinit var maxDetections: EditText
    private lateinit var originalView: ImageView
    private lateinit var resultView: ImageView

    private val picker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri -> uri?.let { selected = it; loadSelected(it) } }
    private val multiPicker = registerForActivityResult(ActivityResultContracts.GetMultipleContents()) { uris -> uris.firstOrNull()?.let { selected = it; loadSelected(it) }; status.text = "${uris.size} imagem(ns) selecionada(s); usando a primeira para a medição visual." }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        status = findViewById(R.id.status); details = findViewById(R.id.details)
        inputWidth = findViewById(R.id.inputWidth); inputHeight = findViewById(R.id.inputHeight)
        confidence = findViewById(R.id.confidence); iou = findViewById(R.id.iou)
        warmup = findViewById(R.id.warmup); repetitions = findViewById(R.id.repetitions); maxDetections = findViewById(R.id.maxDetections)
        originalView = findViewById(R.id.imageOriginal); resultView = findViewById(R.id.imageResult)
        confidence.setText("0.25"); iou.setText("0.45"); warmup.setText("1"); repetitions.setText("10"); maxDetections.setText("100")
        logger = Logger(this)
        try {
            yolo = YoloTflite(this)
            inputWidth.setText(yolo.inputWidth.toString()); inputHeight.setText(yolo.inputHeight.toString())
            status.text = "Modelo carregado: model.tflite | input=${yolo.inputWidth}x${yolo.inputHeight}x${yolo.inputChannels}"
        } catch (e: Exception) {
            status.text = "MODEL_LOAD_ERROR\n${e.message}\n${e.stackTraceToString()}"
            logger.append("MODEL_LOAD_ERROR\n${e.stackTraceToString()}")
        }
        findViewById<Button>(R.id.selectImage).setOnClickListener { picker.launch("image/*") }
        findViewById<Button>(R.id.selectImages).setOnClickListener { multiPicker.launch("image/*") }
        findViewById<Button>(R.id.runDetection).setOnClickListener { detect() }
        findViewById<Button>(R.id.runBenchmark).setOnClickListener { benchmark() }
        findViewById<Button>(R.id.export).setOnClickListener { exportResult() }
    }

    private fun loadSelected(uri: Uri) {
        executor.execute {
            try {
                val bitmap = contentResolver.openInputStream(uri).use { input -> android.graphics.BitmapFactory.decodeStream(input) }
                    ?: error("Não foi possível decodificar a imagem")
                currentBitmap?.let { if (!it.isRecycled) it.recycle() }
                currentBitmap = bitmap
                runOnUiThread { originalView.setImageBitmap(bitmap); status.text = "Imagem: ${bitmap.width}x${bitmap.height} | ${bitmap.byteCount} bytes (estimativa do Bitmap)" }
            } catch (e: Exception) { runOnUiThread { status.text = "IMAGE_LOAD_ERROR\n${e.message}" }; logger.append("IMAGE_LOAD_ERROR\n${e.stackTraceToString()}") }
        }
    }

    private fun detect() {
        val bitmap = currentBitmap ?: return toast("Selecione uma imagem")
        executor.execute {
            try {
                val before = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
                val result = yolo.detect(bitmap, confidenceValue(), iouValue(), maxDetectionsValue())
                val after = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
                val output = drawResults(bitmap, result.detections)
                lastExportBitmap = output
                runOnUiThread {
                    resultView.setImageBitmap(output)
                    details.text = formatResult(bitmap, result, before, after)
                }
                logger.append("DETECTION model=model.tflite image=${bitmap.width}x${bitmap.height} preprocess=${result.preprocessMs} inference=${result.inferenceMs} postprocess=${result.postprocessMs} total=${result.totalMs} detections=${result.detections.size} memoryBefore=$before memoryAfter=$after")
            } catch (e: Exception) { runOnUiThread { status.text = "INFERENCE_ERROR\n${e.message}" }; logger.append("INFERENCE_ERROR\n${e.stackTraceToString()}") }
        }
    }

    private fun benchmark() {
        val bitmap = currentBitmap ?: return toast("Selecione uma imagem")
        executor.execute {
            try {
                val summary = Benchmark(yolo).run(bitmap, warmupValue(), repetitionsValue(), confidenceValue(), iouValue(), maxDetectionsValue())
                runOnUiThread { details.text = String.format(Locale.US, "Warm-up: %d execuções\nBenchmark: %d execuções\n\nPreprocess min/max/média: %.3f / %.3f / %.3f ms\nInference min/max/média: %.3f / %.3f / %.3f ms\nTotal min/max/média: %.3f / %.3f / %.3f ms\nMediana total: %.3f ms\nDesvio padrão total: %.3f ms\nFPS aproximado: %.2f", summary.warmupRuns, summary.benchmarkRuns, summary.preprocessMin, summary.preprocessMax, summary.preprocessMean, summary.inferenceMin, summary.inferenceMax, summary.inferenceMean, summary.totalMin, summary.totalMax, summary.totalMean, summary.totalMedian, summary.totalStdDev, summary.fps) }
                logger.append("BENCHMARK warmup=${summary.warmupRuns} runs=${summary.benchmarkRuns} inferenceMean=${summary.inferenceMean} totalMean=${summary.totalMean} fps=${summary.fps}")
            } catch (e: Exception) { runOnUiThread { status.text = "INFERENCE_ERROR\n${e.message}" }; logger.append("INFERENCE_ERROR\n${e.stackTraceToString()}") }
        }
    }

    private fun drawResults(source: Bitmap, detections: List<YoloResult>): Bitmap {
        val out = source.copy(Bitmap.Config.ARGB_8888, true); val canvas = Canvas(out); val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = maxOf(2f, source.width / 400f) }
        val text = Paint(Paint.ANTI_ALIAS_FLAG).apply { textSize = maxOf(24f, source.width / 30f); style = Paint.Style.FILL }
        detections.forEach { d -> canvas.drawRect(d.left, d.top, d.right, d.bottom, paint); canvas.drawText("${yolo.className(d.classId)} %.3f".format(Locale.US, d.confidence), d.left, maxOf(text.textSize, d.top), text) }
        return out
    }

    private fun formatResult(bitmap: Bitmap, m: PipelineMetrics, before: Long, after: Long) = String.format(Locale.US, "Original: %d x %d\nModel input: %d x %d\nResize: letterbox em memória\nFormato: RGB float32, normalização [0,1]\n\nPreprocess: %.3f ms\nInference: %.3f ms\nPostprocess: %.3f ms\nTotal: %.3f ms\nDetections: %d\nMemória processo (estimativa): antes=%d bytes, depois=%d bytes", bitmap.width, bitmap.height, yolo.inputWidth, yolo.inputHeight, m.preprocessMs, m.inferenceMs, m.postprocessMs, m.totalMs, m.detections.size, before, after)
    private fun exportResult() { lastExportBitmap?.let { bitmap -> val file = File(cacheDir, "yolo_result.png"); file.outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }; toast("Resultado salvo temporariamente em ${file.name}") } ?: toast("Execute uma detecção primeiro") }
    private fun confidenceValue() = confidence.text.toString().toFloatOrNull() ?: .25f
    private fun iouValue() = iou.text.toString().toFloatOrNull() ?: .45f
    private fun warmupValue() = warmup.text.toString().toIntOrNull() ?: 1
    private fun repetitionsValue() = repetitions.text.toString().toIntOrNull() ?: 10
    private fun maxDetectionsValue() = maxDetections.text.toString().toIntOrNull() ?: 100
    private fun toast(text: String) = Toast.makeText(this, text, Toast.LENGTH_SHORT).show()
    override fun onDestroy() { executor.shutdownNow(); if (::yolo.isInitialized) yolo.close(); super.onDestroy() }
}
