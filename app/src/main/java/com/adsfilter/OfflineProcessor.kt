package com.adsfilter

import android.content.Context
import android.os.Build
import android.util.Half
import android.util.Log
import java.io.BufferedOutputStream
import java.io.BufferedWriter
import java.io.DataInputStream
import java.io.File
import java.io.FileOutputStream
import java.io.FileWriter
import java.util.Locale

/**
 * Processes WAV files recorded elsewhere (the laptop's stream recorder) through the phone's own
 * YAMNet, producing the same .f16 and .jsonl a live session would have.
 *
 * The audio itself was measured interchangeable with what the phone captures (r = 0.9986), but
 * embeddings computed on a laptop would come from a different TFLite runtime. Running them here
 * removes that last difference between training data and production, and it means any recording
 * can be reprocessed later if the model or the hop size changes.
 *
 * The WAV is NOT copied. Outputs take the input's basename and are paired with the desktop's own
 * copy of the audio at ingest time - which means the desktop verifier, recomputing each frame's
 * rms from its own WAV and comparing against what the phone wrote, also proves the phone processed
 * exactly the file we think it did.
 *
 * Expects 16 kHz mono 16-bit PCM, as tools/record_stream.sh produces.
 */
object OfflineProcessor {

    private const val TAG = "Offline"
    const val INBOX = "inbox"
    const val OUTBOX = "processed"

    class Result(val name: String, val frames: Long, val seconds: Double, val error: String? = null)

    fun inboxDir(ctx: Context) = File(ctx.getExternalFilesDir(null), INBOX).apply { mkdirs() }
    fun outboxDir(ctx: Context) = File(ctx.getExternalFilesDir(null), OUTBOX).apply { mkdirs() }

    /** Processes every .wav in the inbox. [progress] is called per file as it completes. */
    fun processAll(
        ctx: Context,
        net: Yamnet,
        deleteAfter: Boolean,
        progress: (String) -> Unit
    ): List<Result> {
        val inbox = inboxDir(ctx)
        val outbox = outboxDir(ctx)
        val wavs = inbox.listFiles { f -> f.name.lowercase().endsWith(".wav") }
            ?.sortedBy { it.name } ?: emptyList()

        if (wavs.isEmpty()) {
            progress("inbox is empty: ${inbox.absolutePath}")
            return emptyList()
        }
        progress("processing ${wavs.size} file(s) from inbox")

        val out = ArrayList<Result>()
        for ((i, wav) in wavs.withIndex()) {
            val base = wav.name.removeSuffix(".wav").removeSuffix(".WAV")
            progress("[${i + 1}/${wavs.size}] $base")
            val r = try {
                processOne(wav, base, outbox, net)
            } catch (t: Throwable) {
                Log.e(TAG, "failed on ${wav.name}", t)
                Result(base, 0, 0.0, "${t.javaClass.simpleName}: ${t.message}")
            }
            out.add(r)
            if (r.error != null) {
                progress("   FAILED: ${r.error}")
            } else {
                progress(String.format(Locale.US, "   %.1fs -> %d frames", r.seconds, r.frames))
                if (deleteAfter) wav.delete()
            }
        }
        return out
    }

    private fun processOne(wav: File, base: String, outbox: File, net: Yamnet): Result {
        val hdr = readWavHeader(wav)
        require(hdr.sampleRate == Yamnet.SAMPLE_RATE) {
            "expected ${Yamnet.SAMPLE_RATE} Hz, got ${hdr.sampleRate}"
        }
        require(hdr.channels == 1) { "expected mono, got ${hdr.channels} channels" }
        require(hdr.bits == 16) { "expected 16-bit, got ${hdr.bits}" }

        val embFile = File(outbox, "$base.f16")
        val metaFile = File(outbox, "$base.jsonl")
        val emb = BufferedOutputStream(FileOutputStream(embFile), 1 shl 16)
        val meta = BufferedWriter(FileWriter(metaFile), 1 shl 15)

        var samples = 0L
        var frames = 0L
        var ok = false
        try {
            meta.write(
                """{"type":"session","version":${SessionWriter.FORMAT_VERSION},""" +
                    """"source":"offline","source_file":"${wav.name}",""" +
                    """"sample_rate":${Yamnet.SAMPLE_RATE},"channels":1,""" +
                    """"window":${Yamnet.WINDOW},"hop":${Yamnet.HOP},""" +
                    """"embedding_dim":${Yamnet.EMBEDDING_DIM},""" +
                    """"device":"${Build.MODEL}","android_api":${Build.VERSION.SDK_INT}}"""
            )
            meta.newLine()

            val row = ByteArray(Yamnet.EMBEDDING_DIM * 2)
            val emitter = FrameEmitter(net) { index, embedding, rms, topIdx, topScore ->
                for (i in 0 until Yamnet.EMBEDDING_DIM) {
                    val h = Half.toHalf(embedding[i]).toInt()
                    row[i * 2] = (h and 0xFF).toByte()
                    row[i * 2 + 1] = ((h shr 8) and 0xFF).toByte()
                }
                emb.write(row)

                val sb = StringBuilder(120)
                sb.append("""{"f":""").append(index)
                    .append(""","s":""").append(index * Yamnet.HOP)
                    .append(""","rms":""").append(String.format(Locale.US, "%.3f", rms))
                    .append(""","top":[""")
                for (i in topIdx.indices) { if (i > 0) sb.append(','); sb.append(topIdx[i]) }
                sb.append("""],"p":[""")
                for (i in topScore.indices) {
                    if (i > 0) sb.append(',')
                    sb.append(String.format(Locale.US, "%.3f", topScore[i]))
                }
                sb.append("]}")
                meta.write(sb.toString())
                meta.newLine()
            }

            DataInputStream(wav.inputStream().buffered(1 shl 16)).use { input ->
                input.skipNBytesCompat(hdr.dataOffset.toLong())
                val bytes = ByteArray(Yamnet.HOP * 2)
                val pcm = ShortArray(Yamnet.HOP)
                var remaining = hdr.dataBytes
                while (remaining > 0) {
                    val want = minOf(bytes.size.toLong(), remaining).toInt()
                    val got = input.readNBytesCompat(bytes, want)
                    if (got <= 0) break
                    val n = got / 2
                    for (i in 0 until n) {
                        pcm[i] = ((bytes[i * 2].toInt() and 0xFF) or
                            (bytes[i * 2 + 1].toInt() shl 8)).toShort()
                    }
                    emitter.push(pcm, n)
                    samples += n
                    remaining -= got
                }
            }

            frames = emitter.frames()
            val seconds = samples.toDouble() / Yamnet.SAMPLE_RATE
            meta.write(
                """{"type":"end","frames":$frames,"samples":$samples,""" +
                    """"duration_s":${String.format(Locale.US, "%.3f", seconds)}}"""
            )
            meta.newLine()
            ok = true
            return Result(base, frames, seconds)
        } finally {
            try { emb.flush(); emb.close() } catch (_: Throwable) {}
            try { meta.flush(); meta.close() } catch (_: Throwable) {}
            if (!ok) {
                // A half-written pair would fail verification later in a confusing way.
                embFile.delete(); metaFile.delete()
            }
        }
    }

    // ---- minimal WAV header parsing ----

    class WavHeader(val sampleRate: Int, val channels: Int, val bits: Int,
                    val dataOffset: Int, val dataBytes: Long)

    fun readWavHeader(f: File): WavHeader {
        f.inputStream().use { input ->
            val head = ByteArray(12)
            require(input.read(head) == 12) { "file too short" }
            require(String(head, 0, 4) == "RIFF" && String(head, 8, 4) == "WAVE") {
                "not a RIFF/WAVE file"
            }
            var pos = 12
            var sr = 0; var ch = 0; var bits = 0
            val chunk = ByteArray(8)
            while (true) {
                if (input.read(chunk) != 8) break
                val id = String(chunk, 0, 4)
                val size = le32(chunk, 4)
                pos += 8
                if (id == "fmt ") {
                    val fmt = ByteArray(size)
                    input.read(fmt)
                    ch = le16(fmt, 2)
                    sr = le32(fmt, 4)
                    bits = le16(fmt, 14)
                    pos += size
                } else if (id == "data") {
                    return WavHeader(sr, ch, bits, pos, size.toLong() and 0xFFFFFFFFL)
                } else {
                    input.skip(size.toLong())
                    pos += size
                }
            }
            throw IllegalArgumentException("no data chunk found")
        }
    }

    private fun le16(b: ByteArray, o: Int) = (b[o].toInt() and 0xFF) or ((b[o + 1].toInt() and 0xFF) shl 8)
    private fun le32(b: ByteArray, o: Int) =
        (b[o].toInt() and 0xFF) or ((b[o + 1].toInt() and 0xFF) shl 8) or
            ((b[o + 2].toInt() and 0xFF) shl 16) or ((b[o + 3].toInt() and 0xFF) shl 24)

    private fun java.io.InputStream.skipNBytesCompat(n: Long) {
        var left = n
        while (left > 0) {
            val s = skip(left)
            if (s <= 0) {
                if (read() < 0) return
                left--
            } else left -= s
        }
    }

    private fun java.io.InputStream.readNBytesCompat(buf: ByteArray, want: Int): Int {
        var off = 0
        while (off < want) {
            val r = read(buf, off, want - off)
            if (r < 0) break
            off += r
        }
        return off
    }
}
