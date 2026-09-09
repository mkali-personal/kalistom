package com.adsfilter

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
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
import java.util.Locale
import java.util.concurrent.ConcurrentLinkedQueue
import kotlin.math.log10
import kotlin.math.sqrt

/**
 * Phase 1: the recorder. Captures playback audio, runs YAMNet on it, and writes aligned
 * audio + embedding + metadata sessions to disk. It does NOT mute anything - that is Phase 4.
 *
 * Session lifecycle is driven by whether anything is actually playing. When playback stops for
 * [IDLE_CLOSE_MS] the session is closed; when it resumes a new one opens. Sessions are therefore
 * always contiguous audio, which is what makes "frame k covers samples [k*HOP, k*HOP+WINDOW)"
 * true without qualification.
 */
class RecorderService : Service() {

    companion object {
        private const val TAG = "Recorder"
        private const val CHANNEL_ID = "recorder"
        private const val NOTIF_ID = 1

        const val EXTRA_RESULT_CODE = "result_code"
        const val EXTRA_DATA = "data"
        const val ACTION_STOP = "com.adsfilter.STOP"
        const val ACTION_MARK = "com.adsfilter.MARK"
        const val ACTION_ATTEN_TEST = "com.adsfilter.ATTEN_TEST"

        /** Seconds per stage of the attenuation A/B test. */
        private const val ATTEN_STAGE_SEC = 15L

        /** Close the session after this long with nothing playing. */
        private const val IDLE_CLOSE_MS = 30_000L

        /** Sessions shorter than this are not worth keeping. */
        private const val MIN_SESSION_SEC = 10.0

        /** Delete oldest sessions beyond this budget. ~115 MB/hour of WAV. */
        private const val STORAGE_BUDGET_BYTES = 6L * 1024 * 1024 * 1024

        @Volatile var isRunning = false; private set
        @Volatile var status: String = "idle"; private set

        val logLines = ArrayList<String>()
        var listener: ((String) -> Unit)? = null

        fun log(line: String) {
            Log.i(TAG, line)
            synchronized(logLines) {
                logLines.add(line)
                if (logLines.size > 500) logLines.removeAt(0)
            }
            Handler(Looper.getMainLooper()).post { listener?.invoke(line) }
        }
    }

    private var projection: MediaProjection? = null
    private var record: AudioRecord? = null
    private var worker: Thread? = null
    private var yamnet: Yamnet? = null
    private var session: SessionWriter? = null

    @Volatile private var stopping = false
    private val markQueue = ConcurrentLinkedQueue<String>()
    @Volatile private var attenTestRunning = false

    private lateinit var audioManager: AudioManager
    private val mainHandler = Handler(Looper.getMainLooper())

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> { stopAll(); return START_NOT_STICKY }
            ACTION_MARK -> { markQueue.add("user"); return START_STICKY }
            ACTION_ATTEN_TEST -> { startAttenuationTest(); return START_STICKY }
        }
        if (isRunning) return START_STICKY

        val resultCode = intent?.getIntExtra(EXTRA_RESULT_CODE, 0) ?: 0
        @Suppress("DEPRECATION")
        val data: Intent? = intent?.getParcelableExtra(EXTRA_DATA)
        if (data == null) {
            log("ERROR: no MediaProjection token")
            stopSelf()
            return START_NOT_STICKY
        }

        audioManager = getSystemService(AudioManager::class.java)
        createChannel()
        // Android 14+: the FGS must exist before getMediaProjection() is called.
        startForeground(NOTIF_ID, buildNotification("starting..."), fgsType())

        try {
            start(resultCode, data)
        } catch (t: Throwable) {
            log("ERROR: ${t.javaClass.simpleName}: ${t.message}")
            Log.e(TAG, "start failed", t)
            stopAll()
        }
        return START_STICKY
    }

    private fun fgsType(): Int =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q)
            ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION
        else 0

    private fun start(resultCode: Int, data: Intent) {
        val mpm = getSystemService(MediaProjectionManager::class.java)
        val proj = mpm.getMediaProjection(resultCode, data)
            ?: throw IllegalStateException("getMediaProjection returned null (token already used?)")
        projection = proj
        proj.registerCallback(object : MediaProjection.Callback() {
            override fun onStop() {
                log("MediaProjection stopped by system")
                stopAll()
            }
        }, mainHandler)

        yamnet = Yamnet(this)

        val config = AudioPlaybackCaptureConfiguration.Builder(proj)
            .addMatchingUsage(AudioAttributes.USAGE_MEDIA)
            .addMatchingUsage(AudioAttributes.USAGE_GAME)
            .addMatchingUsage(AudioAttributes.USAGE_UNKNOWN)
            .build()

        // Ask the framework for 16 kHz mono directly: it downmixes and resamples in the audio
        // pipeline, which avoids hand-rolled DSP and any chance of drift between the WAV and the
        // embedding stream. Phase 0 showed the source is dual-mono and rolls off ~10.5 kHz, so
        // this costs ~0.3% of energy.
        val format = AudioFormat.Builder()
            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
            .setSampleRate(Yamnet.SAMPLE_RATE)
            .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
            .build()

        val minBuf = AudioRecord.getMinBufferSize(
            Yamnet.SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
        check(minBuf > 0) { "getMinBufferSize returned $minBuf" }

        val rec = AudioRecord.Builder()
            .setAudioFormat(format)
            .setBufferSizeInBytes(minBuf * 8)
            .setAudioPlaybackCaptureConfig(config)
            .build()
        check(rec.state == AudioRecord.STATE_INITIALIZED) { "AudioRecord state=${rec.state}" }

        log("capture: ${rec.sampleRate} Hz, ${rec.channelCount} ch (requested ${Yamnet.SAMPLE_RATE}/1)")
        if (rec.sampleRate != Yamnet.SAMPLE_RATE || rec.channelCount != 1) {
            log("WARNING: framework gave a different format than requested; frames may be wrong")
        }

        record = rec
        stopping = false
        isRunning = true
        rec.startRecording()

        worker = Thread({ loop(rec) }, "recorder").also { it.start() }
        log("recorder started")
    }

    private fun loop(rec: AudioRecord) {
        val net = yamnet ?: return
        val sessionsDir = File(getExternalFilesDir(null), "sessions")
        enforceStorageBudget(sessionsDir)

        val readBuf = ShortArray(Yamnet.HOP)          // read in exact hop-sized chunks
        // WINDOW is not a multiple of HOP, so the analysis buffer must hold more than one window
        // and advance by exactly HOP per frame. Getting this wrong desynchronises the embeddings
        // from the WAV without any visible symptom.
        val buf = FloatArray(Yamnet.WINDOW + 2 * Yamnet.HOP)
        val window = FloatArray(Yamnet.WINDOW)
        var fill = 0                                  // valid samples currently in `buf`
        var frameIndex = 0L
        var lastActiveMs = System.currentTimeMillis()
        var lastNotifMs = 0L
        var topLine = ""

        while (!stopping) {
            var got = 0
            while (got < readBuf.size && !stopping) {
                val n = rec.read(readBuf, got, readBuf.size - got)
                if (n <= 0) {
                    if (n < 0) log("AudioRecord.read error $n")
                    break
                }
                got += n
            }
            if (stopping) break
            if (got < readBuf.size) continue

            val now = System.currentTimeMillis()
            val rms = rmsDbfs(readBuf, got)
            val playing = audioManager.isMusicActive
            if (playing) lastActiveMs = now

            // --- session lifecycle -------------------------------------------------
            if (session == null && playing) {
                session = SessionWriter(sessionsDir, Yamnet.EMBEDDING_DIM)
                frameIndex = 0
                fill = 0
                log("session start: ${session!!.name}")
            } else if (session != null && !playing && now - lastActiveMs > IDLE_CLOSE_MS) {
                closeSession()
                frameIndex = 0
                fill = 0
                enforceStorageBudget(sessionsDir)
            }

            val s = session
            if (s == null) {
                if (now - lastNotifMs > 2000) {
                    lastNotifMs = now
                    updateNotification("idle - waiting for playback")
                }
                continue
            }

            while (true) {
                val label = markQueue.poll() ?: break
                s.writeMarker(label)
                log("marker '$label' at ${"%.1f".format(Locale.US, s.durationSeconds())}s")
            }

            s.writeAudio(readBuf, got)

            // --- framing -----------------------------------------------------------
            // Append this read, then emit every frame the buffer now completes. Frame k always
            // covers absolute samples [k*HOP, k*HOP + WINDOW) of the WAV.
            for (i in 0 until got) {
                buf[fill + i] = readBuf[i] / 32768.0f
            }
            fill += got

            while (fill >= Yamnet.WINDOW) {
                System.arraycopy(buf, 0, window, 0, Yamnet.WINDOW)
                // rms MUST describe this frame's own window, not the chunk we happened to read
                // last - the desktop verifier recomputes it from the WAV and compares, which is
                // what proves audio and embeddings are aligned.
                val frameRms = rmsDbfs(window)
                try {
                    val r = net.run(window)
                    val top = topK(r.scores, 3)
                    s.writeFrame(frameIndex, r.embedding, frameRms, top.first, top.second)
                    frameIndex++
                    if (net.classNames.isNotEmpty()) {
                        topLine = net.classNames.getOrElse(top.first[0]) { "?" }
                    }
                } catch (t: Throwable) {
                    log("ERROR inference: ${t.javaClass.simpleName}: ${t.message}")
                    Log.e(TAG, "inference failed", t)
                }
                System.arraycopy(buf, Yamnet.HOP, buf, 0, fill - Yamnet.HOP)
                fill -= Yamnet.HOP
            }

            if (now - lastNotifMs > 2000) {
                lastNotifMs = now
                updateNotification(
                    String.format(
                        Locale.US, "%s  %.0f dBFS  %.0fs  %d frames",
                        topLine.ifEmpty { "recording" }, rms, s.durationSeconds(), s.frames()
                    )
                )
            }
        }

        closeSession()
        isRunning = false
    }

    private fun updateNotification(text: String) {
        status = text
        try {
            getSystemService(NotificationManager::class.java)
                .notify(NOTIF_ID, buildNotification(text))
        } catch (_: Throwable) {
        }
    }

    /**
     * Does a session-0 AudioEffect attenuate what AudioPlaybackCapture sees?
     *
     * This is BLOCKING for Phase 4. The volume slider was measured not to affect capture, which is
     * what lets the detector keep watching while it mutes. If the effect DOES attenuate the capture
     * path, that property is lost: muting an ad would feed the model silence and it could never
     * detect the ad ending. Markers are written into the session so the analysis is exact.
     */
    private fun startAttenuationTest() {
        if (attenTestRunning) { log("attenuation test already running"); return }
        if (session == null) { log("start a recording first"); return }
        attenTestRunning = true
        Thread({
            try {
                val stages = listOf<Triple<String, Attenuator.Kind?, Float>>(
                    Triple("baseline", null, 0f),
                    Triple("loudness_-40db", Attenuator.Kind.LOUDNESS, -40f),
                    Triple("off_1", null, 0f),
                    Triple("dynamics_-60db", Attenuator.Kind.DYNAMICS, -60f),
                    Triple("off_2", null, 0f)
                )
                for ((label, kind, db) in stages) {
                    if (kind == null) {
                        Attenuator.releaseQuietly()
                    } else if (!Attenuator.attach(kind, db)) {
                        markQueue.add("${label}_FAILED")
                        log("stage $label: attach refused")
                        continue
                    }
                    markQueue.add(label)
                    log("attenuation stage: $label")
                    Thread.sleep(ATTEN_STAGE_SEC * 1000)
                }
                markQueue.add("atten_test_end")
                log("attenuation test done - stop the recording and run trainer/attenuation_report.py")
            } catch (t: Throwable) {
                log("attenuation test error: ${t.message}")
            } finally {
                Attenuator.releaseQuietly()
                attenTestRunning = false
            }
        }, "atten-test").start()
    }

    private fun closeSession() {
        val s = session ?: return
        session = null
        if (s.durationSeconds() < MIN_SESSION_SEC) {
            log("session too short (${"%.1f".format(Locale.US, s.durationSeconds())}s) - discarded")
            s.discard()
        } else {
            s.close()
            log(
                "session done: ${s.name}  ${"%.0f".format(Locale.US, s.durationSeconds())}s  " +
                    "${s.frames()} frames"
            )
        }
    }

    /** Deletes oldest sessions until the directory fits the budget. */
    private fun enforceStorageBudget(dir: File) {
        val files = dir.listFiles() ?: return
        var total = files.sumOf { it.length() }
        if (total <= STORAGE_BUDGET_BYTES) return

        val names = files.mapNotNull { f ->
            f.name.takeIf { it.startsWith("sess_") }?.substringBeforeLast('.')
        }.distinct().sorted()

        for (n in names) {
            if (total <= STORAGE_BUDGET_BYTES) break
            listOf("$n.wav", "$n.f16", "$n.jsonl").forEach { fn ->
                val f = File(dir, fn)
                if (f.exists()) { total -= f.length(); f.delete() }
            }
            log("rotated out old session $n")
        }
    }

    /** RMS of a normalised float window, in dBFS. */
    private fun rmsDbfs(w: FloatArray): Float {
        var sum = 0.0
        for (v in w) sum += v.toDouble() * v.toDouble()
        val rms = sqrt(sum / w.size)
        return if (rms <= 0.0) -120f
        else (20.0 * log10(rms)).coerceAtLeast(-120.0).toFloat()
    }

    private fun rmsDbfs(buf: ShortArray, n: Int): Float {
        var sum = 0.0
        for (i in 0 until n) {
            val v = buf[i].toDouble()
            sum += v * v
        }
        val rms = sqrt(sum / n)
        return if (rms <= 0.0) -120f
        else (20.0 * log10(rms / 32768.0)).coerceAtLeast(-120.0).toFloat()
    }

    private fun topK(scores: FloatArray, k: Int): Pair<IntArray, FloatArray> {
        val idx = IntArray(k)
        val sc = FloatArray(k)
        for (r in 0 until k) {
            var best = -1
            var bestV = Float.NEGATIVE_INFINITY
            for (i in scores.indices) {
                if (scores[i] > bestV && (0 until r).none { idx[it] == i }) {
                    best = i; bestV = scores[i]
                }
            }
            idx[r] = best.coerceAtLeast(0)
            sc[r] = if (best >= 0) scores[best] else 0f
        }
        return idx to sc
    }

    private fun stopAll() {
        stopping = true
        Attenuator.releaseQuietly()   // never leave the device attenuated
        try { worker?.join(3000) } catch (_: InterruptedException) {}
        worker = null
        closeSession()
        try { record?.stop() } catch (_: Throwable) {}
        record?.release(); record = null
        try { projection?.stop() } catch (_: Throwable) {}
        projection = null
        yamnet?.close(); yamnet = null
        isRunning = false
        status = "stopped"
        log("recorder stopped")
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onDestroy() {
        if (isRunning) stopAll()
        super.onDestroy()
    }

    private fun createChannel() {
        val nm = getSystemService(NotificationManager::class.java)
        if (nm.getNotificationChannel(CHANNEL_ID) == null) {
            nm.createNotificationChannel(
                NotificationChannel(CHANNEL_ID, "Recorder", NotificationManager.IMPORTANCE_LOW)
                    .apply { setShowBadge(false) }
            )
        }
    }

    private fun buildNotification(text: String): Notification {
        val stop = PendingIntent.getService(
            this, 0,
            Intent(this, RecorderService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE
        )
        val mark = PendingIntent.getService(
            this, 1,
            Intent(this, RecorderService::class.java).setAction(ACTION_MARK),
            PendingIntent.FLAG_IMMUTABLE
        )
        val open = PendingIntent.getActivity(
            this, 2,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE
        )
        return Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("ads-filter recorder")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.presence_audio_online)
            .setOngoing(true)
            .setContentIntent(open)
            .addAction(Notification.Action.Builder(null, "Mark", mark).build())
            .addAction(Notification.Action.Builder(null, "Stop", stop).build())
            .build()
    }
}
