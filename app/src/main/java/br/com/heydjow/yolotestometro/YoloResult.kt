package br.com.heydjow.yolotestometro

data class YoloResult(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val classId: Int,
    val confidence: Float
)

data class PipelineMetrics(
    val preprocessMs: Double,
    val inferenceMs: Double,
    val postprocessMs: Double,
    val totalMs: Double,
    val detections: List<YoloResult>
)
