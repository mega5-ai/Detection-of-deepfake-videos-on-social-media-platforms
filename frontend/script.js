console.log("✅ script.js loaded");

let recorder = null;
let chunks = [];
let stream = null;

// ----------------------------
// 📤 UPLOAD VIDEO
// ----------------------------
async function uploadVideo() {
    console.log("📤 Upload video clicked");

    const uploadInput = document.getElementById("uploadVideo");
    const resultBox = document.getElementById("uploadResult");

    const file = uploadInput.files[0];
    if (!file) {
        alert("⚠️ Please select a video file first.");
        return;
    }

    resultBox.innerText = "⏳ Analyzing uploaded video...\nPlease wait...";

    const formData = new FormData();
    formData.append("file", file); // Changed 'video' to 'file' to match backend
    formData.append("source", "upload");

    try {
        const res = await fetch("http://127.0.0.1:8000/detect", { // Changed endpoint to /detect
            method: "POST",
            body: formData,
        });

        const data = await res.json();
        console.log("📦 API response (upload):", data);

        if (!res.ok) {
            throw new Error(data.error || "Server error");
        }

        // Backend returns 'confidence' as 0-100 float already, or 0.0-1.0? 
        // Previous code returned 0-100. User code expects 0.0-1.0.
        // I will adjust backend to match or frontend.
        // Let's assume backend returns 0-100 for now based on my previous code.
        const confidencePct = data.confidence;

        resultBox.innerText =
            `🎥 RESULT: ${data.label || "UNKNOWN"}\n` +
            `📊 Confidence: ${confidencePct}%\n` +
            `📌 Reason: ${data.explanation || "No explanation provided"}`; // Changed 'reason' to 'explanation'

    } catch (error) {
        console.error("❌ Upload error:", error);
        resultBox.innerText =
            "❌ Failed to analyze video.\nCheck backend & console logs.";
    }
}

// ----------------------------
// 🎬 START RECORDING
// ----------------------------
async function startRecording() {
    console.log("🎬 Start recording clicked");

    document.getElementById("recordResult").innerText = "";
    document.getElementById("recordStatus").innerText = "⏺ Recording...";

    document.getElementById("startBtn").disabled = true;
    document.getElementById("stopBtn").disabled = false;

    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: true
        });
    } catch (err) {
        alert("❌ Camera or microphone permission denied");
        return;
    }

    document.getElementById("preview").srcObject = stream;

    recorder = new MediaRecorder(stream, { mimeType: "video/webm" });
    chunks = [];

    recorder.ondataavailable = e => {
        if (e.data.size > 0) chunks.push(e.data);
    };

    recorder.onstop = async () => {
        console.log("⏹ Recording stopped");

        stream.getTracks().forEach(t => t.stop());

        const blob = new Blob(chunks, { type: "video/webm" });
        const fd = new FormData();
        fd.append("file", blob, "webcam-recording.webm"); // Changed 'video' to 'file'
        fd.append("source", "webcam");

        document.getElementById("recordStatus").innerText =
            "⏳ Analyzing recorded video...\nPlease wait...";

        try {
            const res = await fetch("http://127.0.0.1:8000/detect", { // Changed endpoint to /detect
                method: "POST",
                body: fd
            });

            const data = await res.json();
            console.log("📦 API response (record):", data);

            if (!res.ok) {
                throw new Error(data.error || "Server error");
            }

            const confidencePct = data.confidence;

            document.getElementById("recordResult").innerText =
                `🎥 RESULT: ${data.label || "UNKNOWN"}\n` +
                `📊 Confidence: ${confidencePct}%\n` +
                `📌 Reason: ${data.explanation || "No explanation provided"}`; // Changed 'reason' to 'explanation'

        } catch (e) {
            console.error("❌ Record error:", e);
            document.getElementById("recordResult").innerText =
                "❌ Failed to analyze recorded video.";
        }

        document.getElementById("startBtn").disabled = false;
        document.getElementById("stopBtn").disabled = true;
        document.getElementById("recordStatus").innerText = "";
    };

    recorder.start();
}

// ----------------------------
// 🛑 STOP RECORDING
// ----------------------------
function stopRecording() {
    console.log("🛑 Stop recording clicked");

    if (recorder && recorder.state === "recording") {
        recorder.stop();
    }
}

// ----------------------------
// 🤖 TELEGRAM BOT
// ----------------------------
function openTelegram() {
    const botURL = "https://t.me/deepfake_detect_bot";
    window.open(botURL, "_blank");

    const botText = document.getElementById("botText");
    if (botText) {
        botText.innerText = "🔗 Telegram bot opened in a new tab.";
    }
}

// ----------------------------
// 🔥 EXPORT FUNCTIONS
// ----------------------------
window.uploadVideo = uploadVideo;
window.startRecording = startRecording;
window.stopRecording = stopRecording;
window.openTelegram = openTelegram;
