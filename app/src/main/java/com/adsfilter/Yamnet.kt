package com.adsfilter

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * YAMNet (TF Hub TFLite build) wrapper.
 *
 * The model's `waveform` input is DYNAMIC: resizing it to exactly [WINDOW] samples makes the model
 * emit exactly one frame, which puts framing under our control rather than the model's. That
 * matters more than it sounds - the whole labelling pipeline depends on being able to map an
 * Audacity timestamp to an embedding frame index, so the hop must be ours and must be exact.
 *
 * Outputs (one frame each):
 *   0  scores     [1, 521]   AudioSet class scores
 *   1  embedding  [1, 1024]  penultimate layer - what the classifier head trains on
 *   2  melSpec    [1, 64]    log-mel spectrogram
 */
class Yamnet(context: Context) : AutoCloseable {

    companion object {
        private const val TAG = "Yamnet"
        const val MODEL_ASSET = "yamnet.tflite"

        /** 0.975 s at 16 kHz - the canonical YAMNet patch length. */
        const val WINDOW = 15600

        /** 0.48 s at 16 kHz. Frame k covers samples [k*HOP, k*HOP + WINDOW). */
        const val HOP = 7680

        const val SAMPLE_RATE = 16000
        const val EMBEDDING_DIM = 1024
        const val NUM_CLASSES = 521
        const val MEL_BINS = 64
    }

    private val interpreter: Interpreter
    private val scores = Array(1) { FloatArray(NUM_CLASSES) }
    private val embedding = Array(1) { FloatArray(EMBEDDING_DIM) }
    private val melSpec = Array(1) { FloatArray(MEL_BINS) }

    /** Human-readable AudioSet class names, index-aligned with [scores]. */
    val classNames: List<String>

    init {
        val model = loadModel(context)
        val opts = Interpreter.Options().apply {
            numThreads = 2
            // NNAPI is deliberately off: for a model this small the delegate hand-off usually
            // costs more than it saves, and it varies by device. Revisit with measurements.
        }
        interpreter = Interpreter(model, opts)

        // Make the model emit exactly one frame per call.
        interpreter.resizeInput(0, intArrayOf(WINDOW))
        interpreter.allocateTensors()

        classNames = loadClassNames(context)

        val inShape = interpreter.getInputTensor(0).shape().joinToString("x")
        val outShapes = (0 until interpreter.outputTensorCount)
            .joinToString(", ") { interpreter.getOutputTensor(it).shape().joinToString("x") }
        Log.i(TAG, "loaded: in=$inShape out=[$outShapes] classNames=${classNames.size}")

        require(interpreter.outputTensorCount >= 2) {
            "model exposes ${interpreter.outputTensorCount} outputs; need embeddings. " +
                "Wrong YAMNet build? See tools/fetch_model.sh"
        }
    }

    /**
     * Runs one window. [window] must be exactly [WINDOW] samples in -1.0..1.0.
     * Returns a reference to internal buffers - copy before the next call.
     */
    fun run(window: FloatArray): Result {
        require(window.size == WINDOW) { "expected $WINDOW samples, got ${window.size}" }
        val outputs = HashMap<Int, Any>(3)
        outputs[0] = scores
        outputs[1] = embedding
        if (interpreter.outputTensorCount > 2) outputs[2] = melSpec
        interpreter.runForMultipleInputsOutputs(arrayOf<Any>(window), outputs)
        return Result(scores[0], embedding[0], melSpec[0])
    }

    class Result(val scores: FloatArray, val embedding: FloatArray, val melSpec: FloatArray)

    /** Indices of the [n] highest-scoring classes, descending. */
    fun topClasses(scores: FloatArray, n: Int = 3): List<Pair<String, Float>> =
        scores.indices.sortedByDescending { scores[it] }.take(n).map {
            (classNames.getOrElse(it) { "class$it" }) to scores[it]
        }

    override fun close() = interpreter.close()

    private fun loadModel(context: Context): MappedByteBuffer {
        context.assets.openFd(MODEL_ASSET).use { fd ->
            FileInputStream(fd.fileDescriptor).use { input ->
                return input.channel.map(
                    FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength
                )
            }
        }
    }

    private fun loadClassNames(context: Context): List<String> = try {
        context.assets.open("yamnet_class_map.csv").bufferedReader().useLines { lines ->
            // index,mid,display_name - display_name may be quoted and contain commas,
            // e.g. 1,/m/0ytgt,"Child speech, kid speaking"
            lines.drop(1)
                .filter { it.isNotBlank() }
                .map { it.split(",", limit = 3).getOrElse(2) { "" }.trim().trim('"') }
                .toList()
        }
    } catch (t: Throwable) {
        Log.w(TAG, "no class map asset; using numeric labels")
        emptyList()
    }
}
