package com.example.myapplication

// ============================================================================
// Activity Recognition Screen
// - 실시간 센서 신호 표시 (accelerometer + gyroscope)
// - 2초 window, 0.5초 stride 기반 rule-based activity recognition
// - 현재 예측 activity 화면 표시
// - Data Collection Screen (MainActivity.kt)과 완전히 분리된 별도 파일
// ============================================================================

import android.content.Context
import android.content.pm.ActivityInfo
import android.hardware.*
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.*
import androidx.compose.ui.graphics.*
import androidx.compose.ui.text.font.*
import androidx.compose.ui.unit.*
import com.example.myapplication.ui.theme.MyApplicationTheme
import org.apache.commons.math3.transform.*
import java.util.Locale
import kotlin.math.*

class ActivityRecognitionActivity : ComponentActivity(), SensorEventListener {

    // ── 센서 ──
    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null

    // ── 실시간 센서 값 (UI 표시용) ──
    private var accX by mutableStateOf(0f)
    private var accY by mutableStateOf(0f)
    private var accZ by mutableStateOf(0f)
    private var gyroX by mutableStateOf(0f)
    private var gyroY by mutableStateOf(0f)
    private var gyroZ by mutableStateOf(0f)

    // ── Recognition 상태 ──
    private var isRecognizing by mutableStateOf(false)
    private var currentActivity by mutableStateOf("---")
    private var statusMessage by mutableStateOf("Ready")

    // ── Feature 값 (디버그 표시용) ──
    private var featAccMagStd by mutableStateOf(0.0)
    private var featGyroMagStd by mutableStateOf(0.0)
    private var featStdAv by mutableStateOf(0.0)
    private var featJerk by mutableStateOf(0.0)
    private var featStepFq by mutableStateOf(0.0)
    private var featHF by mutableStateOf(0.0)
    private var featFLRS by mutableStateOf(0.0)
    private var featAccXBipolar by mutableStateOf(0.0)
    private var featGyroZBipolar by mutableStateOf(0.0)

    // ── Gravity 추정 (EMA 방식 — Python fallback과 동일) ──
    private var gravX = 0.0
    private var gravY = 0.0
    private var gravZ = 9.81 // 초기 추정값
    private val gravAlpha = 0.02 // EMA alpha (Python _lowpass_ema 기본값과 동일)

    // ── 센서 샘플 버퍼 ──
    data class SensorSample(
        val timestamp: Long, // System.currentTimeMillis()
        val ax: Float, val ay: Float, val az: Float,
        val gx: Float, val gy: Float, val gz: Float,
        val gravX: Double, val gravY: Double, val gravZ: Double
    )

    private val sampleBuffer = mutableListOf<SensorSample>()
    private val windowMs = 2000L  // 2초 window
    private val strideMs = 500L   // 0.5초 stride
    private var lastClassificationTime = 0L
    private var bufferStartTime = 0L

    // ── Lifecycle ──

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        enableEdgeToEdge()

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        setContent {
            MyApplicationTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    ActivityRecognitionScreen(Modifier.padding(innerPadding))
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        accelerometer?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        gyroscope?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
    }

    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(this)
        if (isRecognizing) stopRecognition()
    }

    // ── SensorEventListener ──

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                accX = event.values[0]
                accY = event.values[1]
                accZ = event.values[2]
                // Gravity 추정 갱신 (EMA)
                gravX = gravAlpha * accX + (1 - gravAlpha) * gravX
                gravY = gravAlpha * accY + (1 - gravAlpha) * gravY
                gravZ = gravAlpha * accZ + (1 - gravAlpha) * gravZ

                if (isRecognizing) {
                    addSampleAndClassify()
                }
            }
            Sensor.TYPE_GYROSCOPE -> {
                gyroX = event.values[0]
                gyroY = event.values[1]
                gyroZ = event.values[2]
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    // ====================================================================
    // Recognition 엔진
    // ====================================================================

    private fun startRecognition() {
        sampleBuffer.clear()
        bufferStartTime = System.currentTimeMillis()
        lastClassificationTime = 0L
        currentActivity = "---"
        // Gravity를 현재 가속도계 값으로 초기화
        gravX = accX.toDouble()
        gravY = accY.toDouble()
        gravZ = accZ.toDouble()
        isRecognizing = true
        statusMessage = "Collecting initial window..."
    }

    private fun stopRecognition() {
        isRecognizing = false
        sampleBuffer.clear()
        statusMessage = "Stopped"
    }

    /**
     * 센서 샘플을 버퍼에 추가하고, stride 간격마다 분류를 수행한다.
     * - 2초 window가 안 찼으면 "collecting..." 상태 표시
     * - 0.5초마다 직전 2초 window의 feature를 계산 → classify
     */
    private fun addSampleAndClassify() {
        val now = System.currentTimeMillis()

        sampleBuffer.add(
            SensorSample(
                timestamp = now,
                ax = accX, ay = accY, az = accZ,
                gx = gyroX, gy = gyroY, gz = gyroZ,
                gravX = gravX, gravY = gravY, gravZ = gravZ
            )
        )

        // 오래된 샘플 제거 (window + margin)
        val cutoff = now - windowMs - 1000
        sampleBuffer.removeAll { it.timestamp < cutoff }

        // 버퍼가 2초 미만이면 대기
        val bufferDuration = now - bufferStartTime
        if (bufferDuration < windowMs) {
            statusMessage = "Collecting initial window... (${bufferDuration / 1000}s / 2s)"
            return
        }

        // stride 간격 체크
        if (now - lastClassificationTime < strideMs) return
        lastClassificationTime = now

        // 최근 windowMs 내의 샘플 추출
        val windowStart = now - windowMs
        val windowSamples = sampleBuffer.filter { it.timestamp >= windowStart }

        if (windowSamples.size < 10) {
            statusMessage = "Insufficient samples in window"
            return
        }

        // Feature 계산 → 분류
        val features = computeFeatures(windowSamples)
        currentActivity = classify(features)

        // UI 갱신
        featAccMagStd = features.accMagStd
        featGyroMagStd = features.gyroMagStd
        featStdAv = features.stdAv
        featJerk = features.jerk
        featStepFq = features.stepFq
        featHF = features.hf
        featFLRS = features.fLrs
        featAccXBipolar = features.accXBipolarAmp
        featGyroZBipolar = features.gyroZBipolarAmp

        statusMessage = "Recognizing... (${windowSamples.size} samples in window)"
    }

    // ====================================================================
    // Feature 계산
    // ====================================================================

    data class Features(
        val accMagStd: Double,   // Acc-Mag std
        val gyroMagStd: Double,  // Gyro-Mag std
        val stdAv: Double,       // std A_v (vertical dynamic accel std)
        val jerk: Double,        // Jerk (J_v) = mean(|d(a_v)/dt|)
        val stepFq: Double,      // step_fq (FFT peak in 0.5-4Hz)
        val hf: Double,          // HF = R_HF (high-freq energy ratio)
        val fLrs: Double,        // F_LRS (lateral rotation smoothness)
        // Bipolar amplitude: waving시 같은 window 안에 큰 +/− peak이 함께 나타나는
        // 좌우 왕복 패턴을 잡기 위한 feature (min(max, abs(min)))
        val accXBipolarAmp: Double,
        val gyroZBipolarAmp: Double
    )

    private fun computeFeatures(samples: List<SensorSample>): Features {
        val n = samples.size

        // ── 1. Acc-Mag std ──
        // acc_mag = sqrt(ax² + ay² + az²), 2초 window 내 std
        val accMags = DoubleArray(n) { i ->
            val s = samples[i]
            sqrt((s.ax * s.ax + s.ay * s.ay + s.az * s.az).toDouble())
        }
        val accMagStd = std(accMags)

        // ── 2. Gyro-Mag std ──
        // gyro_mag = sqrt(gx² + gy² + gz²), 2초 window 내 std
        val gyroMags = DoubleArray(n) { i ->
            val s = samples[i]
            sqrt((s.gx * s.gx + s.gy * s.gy + s.gz * s.gz).toDouble())
        }
        val gyroMagStd = std(gyroMags)

        // ── 3. Vertical dynamic acceleration (a_v) ──
        // Python과 동일 방식:
        //   gravity = EMA of raw acc → gravity unit vector
        //   dynamic accel = raw - gravity
        //   a_v = dot(dynamic_accel, gravity_unit_vector)
        val avValues = DoubleArray(n) { i ->
            val s = samples[i]
            val gMag = sqrt(s.gravX * s.gravX + s.gravY * s.gravY + s.gravZ * s.gravZ)
            if (gMag < 1e-6) 0.0
            else {
                val gHatX = s.gravX / gMag
                val gHatY = s.gravY / gMag
                val gHatZ = s.gravZ / gMag
                val dynX = s.ax - s.gravX
                val dynY = s.ay - s.gravY
                val dynZ = s.az - s.gravZ
                dynX * gHatX + dynY * gHatY + dynZ * gHatZ
            }
        }
        val stdAv = std(avValues)

        // ── 4. Jerk (J_v) = mean(|d(a_v)/dt|) ──
        // Python 코드: dav / dt, 시간 기반 미분
        var jerkSum = 0.0
        var jerkCount = 0
        for (i in 1 until n) {
            val dtMs = (samples[i].timestamp - samples[i - 1].timestamp).toDouble()
            if (dtMs > 0) {
                val dtSec = dtMs / 1000.0
                jerkSum += abs(avValues[i] - avValues[i - 1]) / dtSec
                jerkCount++
            }
        }
        val jerk = if (jerkCount > 0) jerkSum / jerkCount else 0.0

        // ── 샘플링 레이트 추정 ──
        val totalTimeMs = (samples.last().timestamp - samples.first().timestamp).toDouble()
        val fs = if (totalTimeMs > 0) (n - 1) * 1000.0 / totalTimeMs else 50.0

        // ── 5. step_fq & 6. HF (R_HF): FFT 기반 (a_v 신호 사용) ──
        // Python과 동일: rfft → power spectrum → peak / energy ratio
        val avMean = avValues.average()
        val avCentered = DoubleArray(n) { avValues[it] - avMean }

        var stepFq = 0.0
        var hfRatio = 0.0

        if (n >= 8) {
            val fftResult = rfft(avCentered)
            val power = DoubleArray(fftResult.size) { i ->
                fftResult[i].first * fftResult[i].first + fftResult[i].second * fftResult[i].second
            }
            val freqs = DoubleArray(fftResult.size) { i -> i * fs / n }

            // step_freq: 0.5~4.0 Hz 범위 피크 주파수
            var maxPower = -1.0
            var peakFreq = 0.0
            for (i in power.indices) {
                if (freqs[i] in 0.5..4.0 && power[i] > maxPower) {
                    maxPower = power[i]
                    peakFreq = freqs[i]
                }
            }
            stepFq = peakFreq

            // R_HF: 고주파(>3Hz) 에너지 / 전체 에너지 (DC 제외)
            var totalPower = 0.0
            var hfPower = 0.0
            for (i in 1 until power.size) {
                totalPower += power[i]
                if (freqs[i] > 3.0) hfPower += power[i]
            }
            hfRatio = if (totalPower > 0) hfPower / totalPower else 0.0
        }

        // ── 7. F_LRS: Lateral Rotation Smoothness (gyro_z 기반) ──
        // Python 코드와 동일 공식:
        //   A_z = RMS(gyro_z_centered)
        //   E_L = low band energy (0.3~2.0 Hz)
        //   E_H = high band energy (3.0~8.0 Hz)
        //   K_z = kurtosis(gyro_z_centered)
        //   F_LRS = A_z * (E_L / (E_H + eps)) * (1 / (K_z + eps))
        val eps = 1e-10
        val gzRaw = DoubleArray(n) { samples[it].gz.toDouble() }
        val gzMean = gzRaw.average()
        val gzCentered = DoubleArray(n) { gzRaw[it] - gzMean }

        val azRms = sqrt(gzCentered.map { it * it }.average())

        var fLrs = 0.0
        if (n >= 8) {
            val gzFft = rfft(gzCentered)
            val gzPower = DoubleArray(gzFft.size) { i ->
                gzFft[i].first * gzFft[i].first + gzFft[i].second * gzFft[i].second
            }
            val gzFreqs = DoubleArray(gzFft.size) { i -> i * fs / n }

            var eLow = 0.0
            var eHigh = 0.0
            for (i in gzPower.indices) {
                if (gzFreqs[i] in 0.3..2.0) eLow += gzPower[i]
                if (gzFreqs[i] in 3.0..8.0) eHigh += gzPower[i]
            }

            val m2 = gzCentered.map { it * it }.average()
            val m4 = gzCentered.map { it.pow(4) }.average()
            val kz = m4 / (m2 * m2 + eps)

            fLrs = azRms * (eLow / (eHigh + eps)) * (1.0 / (kz + eps))
        }

        // ── 8. Bipolar Amplitudes (waving 검출용) ──
        // 하나의 window 안에서 acc_x / gyro_z 가 양방향으로 모두 크게 흔들렸는지 측정
        val axRaw = DoubleArray(n) { samples[it].ax.toDouble() }
        val accXBipolarAmp = min(axRaw.max(), abs(axRaw.min()))
        val gyroZBipolarAmp = min(gzRaw.max(), abs(gzRaw.min()))

        return Features(
            accMagStd = accMagStd,
            gyroMagStd = gyroMagStd,
            stdAv = stdAv,
            jerk = jerk,
            stepFq = stepFq,
            hf = hfRatio,
            fLrs = fLrs,
            accXBipolarAmp = accXBipolarAmp,
            gyroZBipolarAmp = gyroZBipolarAmp
        )
    }

    // ====================================================================
    // 분류 트리 (rule-based)
    // 순서: standing → waving → running → walking → stairs_up/stairs_down
    // ====================================================================

    private fun classify(f: Features): String {
        // 1. STANDING: 모든 센서가 거의 정지 상태
        //     Acc_Mag std < 1.5 && Gyro_Mag std < 1 && std A_v < 1 && Jerk < 40

        if (f.accMagStd < 1.5 && f.gyroMagStd < 1 && f.stdAv < 1 && f.jerk < 40) {
            return "STANDING"
        }
        // 2. WAVING: 양방향 x축 가속도 + z축 각속도 큰 좌우 왕복 패턴
        if (f.accXBipolarAmp > 15 && f.gyroZBipolarAmp > 3 && f.gyroMagStd > 1) {
            return "WAVING"
        }
        // 3. RUNNING: 높은 주파수, 높은 가속/자이로 변동
        //    step_fq > 2.0, Acc-Mag std > 6, Gyro-Mag std > 0.4, std A_v > 7.0
        if (f.stepFq > 2.0 && f.accMagStd > 6 && f.gyroMagStd > 0.4 && f.stdAv > 7.0) {
            return "RUNNING"
        }
        // 4. STAIRS UP: F_LRS와 HF로 구분
        if (f.fLrs > 4 && f.hf < 0.6) {
            return "STAIRS_UP"
        }
        // 4. WALKING / STAIRS DOWN: 비교적 낮은 변동
        //    Gyro-Mag std < 0.8, std A_v < 5, Jerk < 130 
        if (f.gyroMagStd < 0.8 && f.stdAv < 5 && f.jerk < 130) {
            return "WALKING"
        }
        //    F_LRS > 4 && HF < 0.6 → STAIRS_UP, 아니면 STAIRS_DOWN
        return "STAIRS_DOWN"
    }

    // ====================================================================
    // 유틸리티 함수
    // ====================================================================

    /** 표준편차 (population std) */
    private fun std(values: DoubleArray): Double {
        if (values.isEmpty()) return 0.0
        val mean = values.average()
        val variance = values.map { (it - mean).pow(2) }.average()
        return sqrt(variance)
    }

    /** 실수 신호에 대한 FFT (Apache Commons Math 사용, N/2+1개 주파수 성분 반환) */
    private fun rfft(signal: DoubleArray): Array<Pair<Double, Double>> {
        val origLen = signal.size
        // Apache Commons FFT는 2의 거듭제곱 길이 필요 → zero-padding
        var pow2 = 1
        while (pow2 < origLen) pow2 *= 2
        val input = signal.copyOf(pow2)

        val transformer = FastFourierTransformer(DftNormalization.STANDARD)
        val result = transformer.transform(input, TransformType.FORWARD)

        val numOutput = origLen / 2 + 1
        return Array(numOutput) { k ->
            Pair(result[k].real, result[k].imaginary)
        }
    }

    // ====================================================================
    // UI — Activity Recognition Screen
    // ====================================================================

    @Composable
    fun ActivityRecognitionScreen(modifier: Modifier = Modifier) {
        Column(
            modifier = modifier
                .fillMaxSize()
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            // ── 제목 ──
            Text("Activity Recognition", style = MaterialTheme.typography.headlineMedium)

            HorizontalDivider()

            // ── 현재 센서 값 표시 ──
            Text("Accelerometer (m/s²)", style = MaterialTheme.typography.titleSmall)
            Text(
                String.format(Locale.US, "X: %.4f   Y: %.4f   Z: %.4f", accX, accY, accZ),
                fontSize = 14.sp
            )

            Text("Gyroscope (rad/s)", style = MaterialTheme.typography.titleSmall)
            Text(
                String.format(Locale.US, "X: %.4f   Y: %.4f   Z: %.4f", gyroX, gyroY, gyroZ),
                fontSize = 14.sp
            )

            HorizontalDivider()

            // ── 상태 표시 ──
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    if (isRecognizing) "● RECOGNIZING" else "○ IDLE",
                    color = if (isRecognizing) Color.Red else Color.Gray,
                    fontSize = 16.sp
                )
                Spacer(Modifier.width(12.dp))
                Text(statusMessage, fontSize = 13.sp, color = Color.Gray)
            }

            // ── 현재 예측 Activity (크게 표시) ──
            Text(
                currentActivity,
                fontSize = 48.sp,
                fontWeight = FontWeight.Bold,
                color = when (currentActivity) {
                    "STANDING" -> Color.Blue
                    "WAVING" -> Color(0xFFFF6F00)
                    "WALKING" -> Color.Green
                    "RUNNING" -> Color.Red
                    "STAIRS_UP" -> Color.Yellow
                    "STAIRS_DOWN" -> Color.Magenta
                    else -> Color.Gray
                },
                modifier = Modifier
                    .align(Alignment.CenterHorizontally)
                    .padding(vertical = 16.dp)
            )

            HorizontalDivider()

            // ── Feature 값 (디버깅용) ──
            Text("Features (current window)", style = MaterialTheme.typography.titleSmall)
            val featureText = String.format(
                Locale.US,
                """
                Acc-Mag std:      %.4f
                Gyro-Mag std:     %.4f
                std A_v:          %.4f
                Jerk:             %.4f
                step_fq:          %.4f
                HF:               %.4f
                F_LRS:            %.4f
                Acc-X Bipolar:    %.4f
                Gyro-Z Bipolar:   %.4f
                """,
                featAccMagStd, featGyroMagStd, featStdAv,
                featJerk, featStepFq, featHF, featFLRS,
                featAccXBipolar, featGyroZBipolar
            )
            Text(featureText, fontSize = 13.sp, fontFamily = FontFamily.Monospace)

            HorizontalDivider()

            // ── Start / Stop 버튼 ──
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Button(
                    onClick = { startRecognition() },
                    enabled = !isRecognizing,
                    modifier = Modifier.weight(1f)
                ) { Text("Start Recognition") }
                Button(
                    onClick = { stopRecognition() },
                    enabled = isRecognizing,
                    modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.error
                    )
                ) { Text("Stop Recognition") }
            }

            Spacer(Modifier.padding(bottom = 16.dp))
        }
    }
}
