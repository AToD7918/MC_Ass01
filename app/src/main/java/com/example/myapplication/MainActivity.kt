package com.example.myapplication

import android.content.Context
import android.content.Intent
import android.content.pm.ActivityInfo
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.myapplication.ui.theme.MyApplicationTheme
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

// 센서 데이터 수집 및 CSV 저장을 담당하는 메인 액티비티
// SensorEventListener를 구현하여 가속도계/자이로스코프 이벤트를 직접 수신
class MainActivity : ComponentActivity(), SensorEventListener {

    // 센서 매니저 및 센서 객체
    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null

    // 가속도계 실시간 값 (Compose UI 갱신용 mutableState)
    private var accX by mutableStateOf(0f)
    private var accY by mutableStateOf(0f)
    private var accZ by mutableStateOf(0f)
    // 자이로스코프 실시간 값
    private var gyroX by mutableStateOf(0f)
    private var gyroY by mutableStateOf(0f)
    private var gyroZ by mutableStateOf(0f)

    // 녹화 상태 및 세션 정보
    private var isRecording by mutableStateOf(false)
    private var selectedLabel by mutableStateOf("standing_still")
    private var sessionId by mutableStateOf("")
    private var recordingStartTime by mutableStateOf(0L)
    private var sampleCount by mutableStateOf(0)
    private var elapsedSeconds by mutableStateOf(0)
    private var lastSavedFile by mutableStateOf("")
    private var statusMessage by mutableStateOf("Ready")

    // CSV 파일 쓰기 객체
    private var csvWriter: BufferedWriter? = null
    private var csvFile: File? = null

    // 그래프용 최근 샘플 히스토리 (최대 graphSize개 유지)
    private val graphSize = 100
    private var accHistory by mutableStateOf(listOf<Triple<Float, Float, Float>>())
    private var gyroHistory by mutableStateOf(listOf<Triple<Float, Float, Float>>())

    // 실험 고정 메타데이터 (왼손, 화면 몸쪽 방향)
    private val phoneHand = "left_hand"
    private val screenDirection = "screen_toward_body"

    // 수집 대상 활동 라벨 목록
    private val labels = listOf(
        "standing_still", "walking", "running", "stairs_up", "stairs_down"
    )

    // 액티비티 생성 시 센서 초기화 및 Compose UI 설정
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        enableEdgeToEdge()

        // 모든 파일 접근 권한 확인 (공용 폴더 저장에 필요)
        if (!Environment.isExternalStorageManager()) {
            startActivity(Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION))
        }

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        setContent {
            MyApplicationTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    DataCollectionScreen(Modifier.padding(innerPadding))
                }
            }
        }
    }

    // 화면 복귀 시 센서 리스너 등록 (SENSOR_DELAY_GAME 주기)
    override fun onResume() {
        super.onResume()
        accelerometer?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        gyroscope?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
    }

    // 화면 이탈 시 센서 해제 및 녹화 중이면 자동 중지
    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(this)
        if (isRecording) stopRecording()
    }

    // 센서 값 수신 콜백 — 가속도계/자이로스코프 값 갱신 및 히스토리 누적
    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                accX = event.values[0]
                accY = event.values[1]
                accZ = event.values[2]
                accHistory = (accHistory + Triple(accX, accY, accZ)).takeLast(graphSize)
                if (isRecording) writeCsvRow()
            }
            Sensor.TYPE_GYROSCOPE -> {
                gyroX = event.values[0]
                gyroY = event.values[1]
                gyroZ = event.values[2]
                gyroHistory = (gyroHistory + Triple(gyroX, gyroY, gyroZ)).takeLast(graphSize)
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    // 현재 센서 값 한 줄을 CSV에 기록
    private fun writeCsvRow() {
        val now = System.currentTimeMillis()
        val elapsed = now - recordingStartTime
        elapsedSeconds = (elapsed / 1000).toInt()
        csvWriter?.let { w ->
            w.write(
                String.format(
                    Locale.US,
                    "%d,%d,%s,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
                    now, elapsed, selectedLabel, sessionId,
                    accX, accY, accZ, gyroX, gyroY, gyroZ
                )
            )
            w.newLine()
            sampleCount++
        }
    }

    // 녹화 시작 — CSV 파일 생성 및 헤더/메타데이터 작성
    private fun startRecording() {
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        sessionId = "${selectedLabel}_$timestamp"

        // 공용 저장소의 mobileComputing 폴더에 저장
        val dir = File(Environment.getExternalStorageDirectory(), "mobileComputing")
        if (!dir.exists()) dir.mkdirs()
        csvFile = File(dir, "${sessionId}.csv")

        csvWriter = BufferedWriter(FileWriter(csvFile)).apply {
            write("# phone_hand=$phoneHand\n")
            write("# screen_direction=$screenDirection\n")
            write("# device_model=${Build.MODEL}\n")
            write("# recording_start=$timestamp\n")
            write("timestamp_ms,elapsed_ms,label,session_id,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z\n")
        }

        recordingStartTime = System.currentTimeMillis()
        sampleCount = 0
        elapsedSeconds = 0
        isRecording = true
        statusMessage = "Recording..."
    }

    // 녹화 중지 — 파일 flush/close 후 저장 완료 메시지 표시
    private fun stopRecording() {
        isRecording = false
        try {
            csvWriter?.flush()
            csvWriter?.close()
        } catch (_: Exception) { }
        csvWriter = null
        lastSavedFile = csvFile?.absolutePath ?: ""
        statusMessage = "Saved: ${csvFile?.name} ($sampleCount samples)"
    }

    // 데이터 수집 화면 전체 UI 구성
    @Composable
    fun DataCollectionScreen(modifier: Modifier = Modifier) {
        Column(
            modifier = modifier
                .fillMaxSize()
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Text(
                "Data Collection",
                style = MaterialTheme.typography.headlineMedium
            )

            // 활동 라벨 선택 버튼 (첫 번째 줄: 3개)
            Text("Activity Label:", style = MaterialTheme.typography.titleSmall)
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                labels.take(3).forEach { label ->
                    val isSel = label == selectedLabel
                    if (isSel) {
                        Button(
                            onClick = {},
                            enabled = !isRecording,
                            modifier = Modifier.weight(1f)
                        ) { Text(label, fontSize = 11.sp, maxLines = 1) }
                    } else {
                        OutlinedButton(
                            onClick = { selectedLabel = label },
                            enabled = !isRecording,
                            modifier = Modifier.weight(1f)
                        ) { Text(label, fontSize = 11.sp, maxLines = 1) }
                    }
                }
            }
            // 활동 라벨 선택 버튼 (두 번째 줄: 2개)
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                labels.drop(3).forEach { label ->
                    val isSel = label == selectedLabel
                    if (isSel) {
                        Button(
                            onClick = {},
                            enabled = !isRecording,
                            modifier = Modifier.weight(1f)
                        ) { Text(label, fontSize = 11.sp, maxLines = 1) }
                    } else {
                        OutlinedButton(
                            onClick = { selectedLabel = label },
                            enabled = !isRecording,
                            modifier = Modifier.weight(1f)
                        ) { Text(label, fontSize = 11.sp, maxLines = 1) }
                    }
                }
            }

            HorizontalDivider()

            // 가속도계 실시간 값 및 그래프
            Text("Accelerometer (m/s²)", style = MaterialTheme.typography.titleSmall)
            Text(
                String.format(Locale.US, "X: %.4f   Y: %.4f   Z: %.4f", accX, accY, accZ),
                fontSize = 14.sp
            )
            SensorGraph(
                history = accHistory,
                label = "Accel",
                yRange = 20f
            )

            // 자이로스코프 실시간 값 및 그래프
            Text("Gyroscope (rad/s)", style = MaterialTheme.typography.titleSmall)
            Text(
                String.format(Locale.US, "X: %.4f   Y: %.4f   Z: %.4f", gyroX, gyroY, gyroZ),
                fontSize = 14.sp
            )
            SensorGraph(
                history = gyroHistory,
                label = "Gyro",
                yRange = 10f
            )

            HorizontalDivider()

            // 녹화 상태 표시 (RECORDING / IDLE)
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    if (isRecording) "● RECORDING" else "○ IDLE",
                    color = if (isRecording) Color.Red else Color.Gray,
                    fontSize = 16.sp
                )
                if (isRecording) {
                    Spacer(Modifier.width(16.dp))
                    Text("${elapsedSeconds}s | $sampleCount samples", fontSize = 14.sp)
                }
            }

            if (sessionId.isNotEmpty()) {
                Text("Session: $sessionId", fontSize = 12.sp, color = Color.Gray)
            }

            // Start / Stop 버튼
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Button(
                    onClick = { startRecording() },
                    enabled = !isRecording,
                    modifier = Modifier.weight(1f)
                ) {
                    Text("Start")
                }
                Button(
                    onClick = { stopRecording() },
                    enabled = isRecording,
                    modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.error
                    )
                ) {
                    Text("Stop")
                }
            }

            HorizontalDivider()

            Text(statusMessage, fontSize = 14.sp)
            if (lastSavedFile.isNotEmpty()) {
                Text("Path: $lastSavedFile", fontSize = 11.sp, color = Color.Gray)
            }
        }
    }

    // 센서 X/Y/Z 축 데이터를 Canvas에 라인 그래프로 표시. accelometer와 gyroscope에 공통으로 사용
    @Composable
    fun SensorGraph(
        history: List<Triple<Float, Float, Float>>,
        label: String,
        yRange: Float
    ) {
        val xColor = Color.Red
        val yColor = Color.Green
        val zColor = Color.Blue

        Canvas(
            modifier = Modifier
                .fillMaxWidth()
                .height(120.dp)
        ) {
            val w = size.width
            val h = size.height
            val mid = h / 2f

            // 중앙선
            drawLine(Color.LightGray, Offset(0f, mid), Offset(w, mid))

            if (history.size < 2) return@Canvas

            val step = w / (graphSize - 1).toFloat()
            val offset = graphSize - history.size

            fun drawAxis(extract: (Triple<Float, Float, Float>) -> Float, color: Color) {
                for (i in 1 until history.size) {
                    val x0 = (offset + i - 1) * step
                    val x1 = (offset + i) * step
                    val y0 = mid - (extract(history[i - 1]) / yRange) * mid
                    val y1 = mid - (extract(history[i]) / yRange) * mid
                    drawLine(color, Offset(x0, y0), Offset(x1, y1), strokeWidth = 2f)
                }
            }

            drawAxis({ it.first }, xColor)
            drawAxis({ it.second }, yColor)
            drawAxis({ it.third }, zColor)
        }

        // 범례
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Text("— X", color = Color.Red, fontSize = 11.sp)
            Text("— Y", color = Color.Green, fontSize = 11.sp)
            Text("— Z", color = Color.Blue, fontSize = 11.sp)
        }
    }
}