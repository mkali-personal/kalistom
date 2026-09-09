package com.adsfilter

import android.os.Build
import android.util.Half
import android.util.Log
import java.io.BufferedOutputStream
import java.io.BufferedWriter
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

/**
 * Writes one contiguous recording session as three aligned files:
 *
 *   sess_<ts>.wav    16 kHz mono 16-bit PCM  - what you open in Audacity
 *   sess_<ts>.f16    float16 embeddings, EMBEDDING_DIM per frame, sequential
 *   sess_<ts>.jsonl  per-frame metadata + markers + session header/footer
 *
 * ALIGNMENT IS THE WHOLE POINT. Frame k covers samples [k*HOP, k*HOP + WINDOW) of the WAV,
 * counting from the first sample in the file. A session is only ever contiguous audio - if
 * playback stops, the session is CLOSED and a new one opened rather than leaving a gap, because
 * a gap would silently shift every subsequent label. See docs/phase0-results.md.
 *
 * Embeddings are written from the device rather than recomputed on the desktop so the classifier
 * trains on exactly the vectors it will see at inference.
 */
class SessionWriter(dir: File, private val embeddingDim: Int) : AutoCloseable {

    companion object {
        private const val TAG = "SessionWriter"
        private const val SAMPLE_RATE = Yamnet.SAMPLE_RATE
        private const val CHANNELS = 1
        private const val BITS = 16
        const val FORMAT_VERSION = 1
    }

    val name: String
    val wavFile: File
    val embFile: File
    val metaFile: File

    private val wav: RandomAccessFile
    private val emb: BufferedOutputStream
    private val meta: BufferedWriter

    private var samplesWritten = 0L
    private var framesWritten = 0L
    private var closed = false

    /** Wall-clock start, so a session can be lined up against anything else that was logged. */
    private val startedUtc: String

    init {
        dir.mkdirs()
        val iso = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US)
            .apply { timeZone = TimeZone.getTimeZone("UTC") }
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        startedUtc = iso.format(Date())

        name = "sess_$stamp"
        wavFile = File(dir, "$name.wav")
        embFile = File(dir, "$name.f16")
        metaFile = File(dir, "$name.jsonl")

        wav = RandomAccessFile(wavFile, "rw")
        writeWavHeader(0)
        emb = BufferedOutputStream(FileOutputStream(embFile), 1 shl 16)
        meta = BufferedWriter(java.io.FileWriter(metaFile), 1 shl 15)

        meta.write(
            """{"type":"session","version":$FORMAT_VERSION,"started_utc":"$startedUtc",""" +
                """"sample_rate":$SAMPLE_RATE,"channels":$CHANNELS,""" +
                """"window":${Yamnet.WINDOW},"hop":${Yamnet.HOP},"embedding_dim":$embeddingDim,""" +
                """"device":"${Build.MODEL}","android_api":${Build.VERSION.SDK_INT}}"""
        )
        meta.newLine()
        Log.i(TAG, "session opened: $name")
    }

    /** Appends mono PCM. [n] samples of [pcm] are written. */
    fun writeAudio(pcm: ShortArray, n: Int) {
        if (closed) return
        val bytes = ByteArray(n * 2)
        for (i in 0 until n) {
            val s = pcm[i].toInt()
            bytes[i * 2] = (s and 0xFF).toByte()
            bytes[i * 2 + 1] = ((s shr 8) and 0xFF).toByte()
        }
        wav.write(bytes)
        samplesWritten += n
    }

    /**
     * Appends one frame's embedding plus its metadata row.
     * [frameIndex] must equal the number of frames already written - it is asserted, not trusted,
     * because a silently dropped frame would corrupt every label after it.
     */
    fun writeFrame(
        frameIndex: Long,
        embedding: FloatArray,
        rmsDbfs: Float,
        topIdx: IntArray,
        topScore: FloatArray
    ) {
        if (closed) return
        check(frameIndex == framesWritten) {
            "frame index skew: got $frameIndex, expected $framesWritten - embeddings would " +
                "no longer line up with the audio"
        }

        val out = ByteArray(embeddingDim * 2)
        for (i in 0 until embeddingDim) {
            val h = Half.toHalf(embedding[i]).toInt()
            out[i * 2] = (h and 0xFF).toByte()
            out[i * 2 + 1] = ((h shr 8) and 0xFF).toByte()
        }
        emb.write(out)

        val startSample = frameIndex * Yamnet.HOP
        val sb = StringBuilder(120)
        sb.append("""{"f":""").append(frameIndex)
            .append(""","s":""").append(startSample)
            .append(""","rms":""").append(fmt(rmsDbfs))
            .append(""","top":[""")
        for (i in topIdx.indices) {
            if (i > 0) sb.append(',')
            sb.append(topIdx[i])
        }
        sb.append("""],"p":[""")
        for (i in topScore.indices) {
            if (i > 0) sb.append(',')
            sb.append(fmt(topScore[i]))
        }
        sb.append("]}")
        meta.write(sb.toString())
        meta.newLine()

        framesWritten++
        if (framesWritten % 64L == 0L) flush()
    }

    /** Records a user marker at the current position - used by the clap alignment test. */
    fun writeMarker(label: String) {
        if (closed) return
        val t = samplesWritten.toDouble() / SAMPLE_RATE
        meta.write(
            """{"type":"marker","sample":$samplesWritten,"t":${fmt(t.toFloat())},""" +
                """"label":"${label.replace("\"", "'")}"}"""
        )
        meta.newLine()
        flush()
        Log.i(TAG, "marker '$label' at ${fmt(t.toFloat())}s")
    }

    fun durationSeconds(): Double = samplesWritten.toDouble() / SAMPLE_RATE
    fun frames(): Long = framesWritten
    fun samples(): Long = samplesWritten

    private fun flush() {
        try {
            emb.flush()
            meta.flush()
        } catch (t: Throwable) {
            Log.w(TAG, "flush failed: ${t.message}")
        }
    }

    override fun close() {
        if (closed) return
        closed = true
        try {
            meta.write(
                """{"type":"end","frames":$framesWritten,"samples":$samplesWritten,""" +
                    """"duration_s":${fmt(durationSeconds().toFloat())}}"""
            )
            meta.newLine()
            meta.flush(); meta.close()
            emb.flush(); emb.close()
            patchWavHeader()
            wav.close()
            Log.i(
                TAG,
                "session closed: $name  ${fmt(durationSeconds().toFloat())}s  " +
                    "$framesWritten frames  ${wavFile.length() / 1024} KB"
            )
        } catch (t: Throwable) {
            Log.e(TAG, "close failed: ${t.message}", t)
        }
    }

    /** Deletes a session that is too short to be worth keeping. */
    fun discard() {
        close()
        listOf(wavFile, embFile, metaFile).forEach { runCatching { it.delete() } }
        Log.i(TAG, "session discarded: $name")
    }

    private fun fmt(v: Float): String = String.format(Locale.US, "%.3f", v)

    // ---- WAV ----

    private fun writeWavHeader(dataLen: Long) {
        val byteRate = SAMPLE_RATE * CHANNELS * BITS / 8
        wav.seek(0)
        wav.writeBytes("RIFF")
        wav.writeInt(Integer.reverseBytes((36 + dataLen).toInt()))
        wav.writeBytes("WAVE")
        wav.writeBytes("fmt ")
        wav.writeInt(Integer.reverseBytes(16))
        wav.writeShort(java.lang.Short.reverseBytes(1.toShort()).toInt())
        wav.writeShort(java.lang.Short.reverseBytes(CHANNELS.toShort()).toInt())
        wav.writeInt(Integer.reverseBytes(SAMPLE_RATE))
        wav.writeInt(Integer.reverseBytes(byteRate))
        wav.writeShort(java.lang.Short.reverseBytes((CHANNELS * BITS / 8).toShort()).toInt())
        wav.writeShort(java.lang.Short.reverseBytes(BITS.toShort()).toInt())
        wav.writeBytes("data")
        wav.writeInt(Integer.reverseBytes(dataLen.toInt()))
    }

    private fun patchWavHeader() {
        val dataLen = samplesWritten * CHANNELS * BITS / 8
        wav.seek(4)
        wav.writeInt(Integer.reverseBytes((36 + dataLen).toInt()))
        wav.seek(40)
        wav.writeInt(Integer.reverseBytes(dataLen.toInt()))
    }
}
