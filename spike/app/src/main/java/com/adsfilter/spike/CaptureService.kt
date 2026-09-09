package com.adsfilter.spike

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioPlaybackCaptureConfiguration
import android.media.AudioRecord
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.Log
import java.io.File
import java.io.RandomAccessFile
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.log10
import kotlin.math.sqrt

/**
 * Phase 0b feasibility spike: taps playback audio via AudioPlaybackCapture, writes a WAV,
 * and reports per-second level so we can tell "captured audio" from "silence = app opted out".
 */
class CaptureService : Service() {

    companion object {
        const val EXTRA_RESULT_CODE = "result_code"
        const val EXTRA_DATA = "data"
        const val EXTRA_LABEL = "label"
        const val ACTION_STOP = "com.adsfilter.spike.STOP"

        private const val TAG = "SpikeCapture"
        private const val CHANNEL_ID = "spike_capture"
        private const val NOTIF_ID = 1

        const val SAMPLE_RATE = 48000
        private const val CHANNELS = 2
        private const val BITS = 16

        /** Frames quieter than this are treated as silence when judging capturability. */
        private const val SILENCE_DBFS = -60.0

        @Volatile
        var isRunning = false
            private set

        val logLines = ArrayList<String>()
        var listener: ((String) -> Unit)? = null

        fun log(line: String) {
            Log.i(TAG, line)
            synchronized(logLines) { logLines.add(line) }
            Handler(Looper.getMainLooper()).post { listener?.invoke(line) }
        }
    }

    private var projection: MediaProjection? = null
    private var record: AudioRecord? = null
    private var worker: Thread? = null

    @Volatile
    private var stopping = false

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopCapture()
            return START_NOT_STICKY
        }

        val resultCode = intent?.getIntExtra(EXTRA_RESULT_CODE, 0) ?: 0
        @Suppress("DEPRECATION")
        val data: Intent? = intent?.getParcelableExtra(EXTRA_DATA)
        val label = intent?.getStringExtra(EXTRA_LABEL)?.ifBlank { "unknown" } ?: "unknown"

        if (data == null) {
            log("ERROR: no MediaProjection result data")
            stopSelf()
            return START_NOT_STICKY
        }

        // Android 14+ requires the FGS (type mediaProjection) to be running BEFORE
        // getMediaProjection() is called. This ordering is the whole point of issue I6.
        createChannel()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                NOTIF_ID,
                buildNotification(label),
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION
            )
        } else {
            startForeground(NOTIF_ID, buildNotification(label))
        }

        try {
            startCapture(resultCode, data, label)
        } catch (t: Throwable) {
            log("ERROR starting capture: " + t.javaClass.simpleName + ": " + t.message)
            stopCapture()
        }
        return START_NOT_STICKY
    }

    private fun startCapture(resultCode: Int, data: Intent, label: String) {
        val mpm = getSystemService(MediaProjectionManager::class.java)
        val proj = mpm.getMediaProjection(resultCode, data)
        if (proj == null) {
            log("ERROR: getMediaProjection returned null (token consumed or denied)")
            stopCapture()
            return
        }
        projection = proj

        // Mandatory on API 34+: register a callback before using the projection.
        proj.registerCallback(object : MediaProjection.Callback() {
            override fun onStop() {
                log("MediaProjection.onStop() fired by system")
                stopCapture()
            }
        }, Handler(Looper.getMainLooper()))

        val config = AudioPlaybackCaptureConfiguration.Builder(proj)
            .addMatchingUsage(AudioAttributes.USAGE_MEDIA)
            .addMatchingUsage(AudioAttributes.USAGE_GAME)
            .addMatchingUsage(AudioAttributes.USAGE_UNKNOWN)
            .build()

        val format = AudioFormat.Builder()
            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
            .setSampleRate(SAMPLE_RATE)
            .setChannelMask(AudioFormat.CHANNEL_IN_STEREO)
            .build()

        val minBuf = AudioRecord.getMinBufferSize(
            SAMPLE_RATE, AudioFormat.CHANNEL_IN_STEREO, AudioFormat.ENCODING_PCM_16BIT
        )
        if (minBuf <= 0) {
            log("ERROR: getMinBufferSize returned " + minBuf)
            stopCapture()
            return
        }

        val rec = AudioRecord.Builder()
            .setAudioFormat(format)
            .setBufferSizeInBytes(minBuf * 8)
            .setAudioPlaybackCaptureConfig(config)
            .build()

        if (rec.state != AudioRecord.STATE_INITIALIZED) {
            log("ERROR: AudioRecord not initialized (state=" + rec.state + ")")
            rec.release()
            stopCapture()
            return
        }

        record = rec
        stopping = false
        isRunning = true

        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val dir = File(getExternalFilesDir(null), "spike")
        dir.mkdirs()
        val wav = File(dir, "cap_" + label + "_" + stamp + ".wav")

        log("--- capture started: " + label + " -> " + wav.name + " ---")
        rec.startRecording()

        val t = Thread({ captureLoop(rec, wav, label) }, "spike-capture")
        worker = t
        t.start()
    }

    private fun captureLoop(rec: AudioRecord, wav: File, label: String) {
        val buf = ShortArray(SAMPLE_RATE / 4 * CHANNELS) // ~250 ms
        var totalFrames = 0L
        var second = 0
        var secSumSq = 0.0
        var secCount = 0L
        var secPeak = 0
        var peakOverall = 0
        var loudSeconds = 0
        var totalSeconds = 0

        RandomAccessFile(wav, "rw").use { raf ->
            writeWavHeader(raf, 0)

            val bytes = ByteArray(buf.size * 2)
            while (!stopping) {
                val n = rec.read(buf, 0, buf.size)
                if (n <= 0) {
                    if (n < 0) log("AudioRecord.read error: " + n)
                    continue
                }
                for (i in 0 until n) {
                    val s = buf[i].toInt()
                    val a = if (s < 0) -s else s
                    if (a > secPeak) secPeak = a
                    if (a > peakOverall) peakOverall = a
                    secSumSq += (s.toDouble() * s.toDouble())
                    bytes[i * 2] = (s and 0xFF).toByte()
                    bytes[i * 2 + 1] = ((s shr 8) and 0xFF).toByte()
                }
                secCount += n
                raf.write(bytes, 0, n * 2)
                totalFrames += (n / CHANNELS)

                // report once per second of captured audio
                if (secCount >= SAMPLE_RATE.toLong() * CHANNELS) {
                    val rms = sqrt(secSumSq / secCount)
                    val dbfs = toDbfs(rms)
                    val peakDb = toDbfs(secPeak.toDouble())
                    totalSeconds++
                    if (dbfs > SILENCE_DBFS) loudSeconds++
                    log(String.format(Locale.US, "t=%3ds  rms=%6.1f dBFS  peak=%6.1f dBFS", second, dbfs, peakDb))
                    second++
                    secSumSq = 0.0
                    secCount = 0
                    secPeak = 0
                }
            }
            raf.channel.force(true)
            patchWavHeader(raf, totalFrames)
        }

        val verdict = when {
            totalSeconds == 0 -> "INCONCLUSIVE (ran < 1 s)"
            loudSeconds == 0 -> "SILENT -> app is very likely opted out of playback capture"
            loudSeconds < totalSeconds / 4 -> "MOSTLY SILENT -> suspicious, re-test with audio definitely playing"
            else -> "AUDIO CAPTURED -> this app is capturable"
        }
        val peakStr = String.format(Locale.US, "%.1f", toDbfs(peakOverall.toDouble()))
        log("--- capture stopped: " + label + " ---")
        log("duration=" + (totalFrames / SAMPLE_RATE) + "s  loud_seconds=" + loudSeconds + "/" + totalSeconds + "  peak=" + peakStr + " dBFS")
        log("VERDICT [" + label + "]: " + verdict)
        log("file: " + wav.absolutePath + " (" + (wav.length() / 1024) + " KB)")

        File(wav.parentFile, wav.nameWithoutExtension + ".verdict.txt").writeText(
            "label=" + label + "\n" +
                "file=" + wav.name + "\n" +
                "duration_s=" + (totalFrames / SAMPLE_RATE) + "\n" +
                "loud_seconds=" + loudSeconds + "\n" +
                "total_seconds=" + totalSeconds + "\n" +
                "peak_dbfs=" + peakStr + "\n" +
                "verdict=" + verdict + "\n"
        )
        isRunning = false
    }

    private fun toDbfs(amplitude: Double): Double =
        if (amplitude <= 0.0) -120.0 else (20.0 * log10(amplitude / 32768.0)).coerceAtLeast(-120.0)

    private fun stopCapture() {
        stopping = true
        try {
            worker?.join(2000)
        } catch (_: InterruptedException) {
        }
        worker = null
        try {
            record?.stop()
        } catch (_: Throwable) {
        }
        record?.release()
        record = null
        try {
            projection?.stop()
        } catch (_: Throwable) {
        }
        projection = null
        isRunning = false
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onDestroy() {
        if (isRunning) stopCapture()
        super.onDestroy()
    }

    // ---- WAV plumbing ----

    private fun writeWavHeader(raf: RandomAccessFile, frames: Long) {
        val byteRate = SAMPLE_RATE * CHANNELS * BITS / 8
        val dataLen = frames * CHANNELS * BITS / 8
        raf.seek(0)
        raf.writeBytes("RIFF")
        raf.writeInt(Integer.reverseBytes((36 + dataLen).toInt()))
        raf.writeBytes("WAVE")
        raf.writeBytes("fmt ")
        raf.writeInt(Integer.reverseBytes(16))
        raf.writeShort(java.lang.Short.reverseBytes(1.toShort()).toInt())
        raf.writeShort(java.lang.Short.reverseBytes(CHANNELS.toShort()).toInt())
        raf.writeInt(Integer.reverseBytes(SAMPLE_RATE))
        raf.writeInt(Integer.reverseBytes(byteRate))
        raf.writeShort(java.lang.Short.reverseBytes((CHANNELS * BITS / 8).toShort()).toInt())
        raf.writeShort(java.lang.Short.reverseBytes(BITS.toShort()).toInt())
        raf.writeBytes("data")
        raf.writeInt(Integer.reverseBytes(dataLen.toInt()))
    }

    private fun patchWavHeader(raf: RandomAccessFile, frames: Long) {
        val dataLen = frames * CHANNELS * BITS / 8
        raf.seek(4)
        raf.writeInt(Integer.reverseBytes((36 + dataLen).toInt()))
        raf.seek(40)
        raf.writeInt(Integer.reverseBytes(dataLen.toInt()))
    }

    // ---- notification ----

    private fun createChannel() {
        val nm = getSystemService(NotificationManager::class.java)
        if (nm.getNotificationChannel(CHANNEL_ID) == null) {
            nm.createNotificationChannel(
                NotificationChannel(CHANNEL_ID, "Spike capture", NotificationManager.IMPORTANCE_LOW)
            )
        }
    }

    private fun buildNotification(label: String): Notification =
        Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("Capturing playback audio")
            .setContentText("target: " + label)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setOngoing(true)
            .build()
}
