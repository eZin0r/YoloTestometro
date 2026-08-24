package br.com.heydjow.yolotestometro

import android.graphics.Bitmap
import kotlin.math.sqrt

class Benchmark(private val yolo: YoloTflite) {
    data class Summary(
        val warmupRuns: Int,
        val benchmarkRuns: Int,
        val preprocessMin: Double,
        val preprocessMax: Double,
        val preprocessMean: Double,
        val inferenceMin: Double,
        val inferenceMax: Double,
        val inferenceMean: Double,
        val totalMin: Double,
        val totalMax: Double,
        val totalMean: Double,
        val totalMedian: Double,
        val totalStdDev: Double,
        val fps: Double
    )

    fun run(bitmap: Bitmap, warmup: Int, repetitions: Int, confidence: Float, iou: Float, maxDetections: Int): Summary {
        repeat(warmup.coerceAtLeast(0)) { yolo.detect(bitmap, confidence, iou, maxDetections) }
        val results = (0 until repetitions.coerceAtLeast(1)).map {
            yolo.detect(bitmap, confidence, iou, maxDetections)
        }
        fun values(selector: (PipelineMetrics) -> Double) = results.map(selector)
        fun min(v: List<Double>) = v.minOrNull() ?: 0.0
        fun max(v: List<Double>) = v.maxOrNull() ?: 0.0
        fun mean(v: List<Double>) = v.average()
        val totals = values { it.totalMs }.sorted()
        val avg = totals.average()
        val variance = totals.map { (it - avg) * (it - avg) }.average()
        val median = if (totals.size % 2 == 0) (totals[totals.size / 2 - 1] + totals[totals.size / 2]) / 2 else totals[totals.size / 2]
        return Summary(
            warmup,
            results.size,
            min(values { it.preprocessMs }), max(values { it.preprocessMs }), mean(values { it.preprocessMs }),
            min(values { it.inferenceMs }), max(values { it.inferenceMs }), mean(values { it.inferenceMs }),
            min(totals), max(totals), avg, median, sqrt(variance), if (avg > 0) 1000.0 / avg else 0.0
        )
    }
}
